import asyncio as aio
import logging
import operator
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import (
    Annotated,
    Optional,
    TypedDict,
    Union,
)

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from multi_objective_debate.agent import AgentGroup, AgentGroups
from multi_objective_debate.config import (
    CHAT_MODEL,
    TEMPREATURE,
)
from multi_objective_debate.model import (
    Divergence,
    Objection,
    Preference,
    Proposal,
    Role,
)
from multi_objective_debate.preset import HHH_PRESET, load_preset
from multi_objective_debate.prompts import (
    DeclareInvocation,
    DeclareResponse,
    ProposeInvocation,
)

logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    filename="logs/multi_objective_debate.log",
    format="[%(asctime)s:%(levelname)s:%(name)s:%(module)s] %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class CollaborateState:
    # 发言顺序
    queue: list[Role]
    # 目前的最优方案
    proposal: Optional[Proposal]
    # 所有未达成一致的方案
    divergences: list[Divergence]


@dataclass
class DebateState:
    pass


@dataclass
class UserFeedbackState:
    proposal: Proposal
    divergences: list[Divergence]


@dataclass
class EndState:
    proposals: list[Proposal]


class Stage(Enum):
    Collaborate = "Collaborate"
    Debate = "Debate"
    UserFeedback = "UserFeedback"
    End = END


class GraphState(TypedDict):
    user_preferences: Annotated[list[Preference], operator.add]

    stage: Stage
    state: Union[CollaborateState, DebateState, UserFeedbackState, EndState]


class MACDConfig(BaseModel):
    pass


async def collaborateRound(
    state: GraphState,
    groups: AgentGroups,
    config: MACDConfig,
) -> GraphState:
    """单轮合作过程"""

    logger.info(f"collaborateRound: {state}")

    async def declare(
        group: AgentGroup,
        invocation: DeclareInvocation,
    ) -> tuple[Role, DeclareResponse]:
        resp = await group.declare.ainvoke(invocation)
        return group.role, resp

    if state["stage"] != Stage.Collaborate:
        raise ValueError(f"Unexpected stage: {state}")

    user_preferences = state["user_preferences"]
    state: CollaborateState = state["state"]

    match state:
        case CollaborateState([], proposal, divergences):
            return {
                "stage": Stage.UserFeedback,
                "state": UserFeedbackState(
                    divergences=divergences,
                    proposal=proposal,
                ),
            }
        case CollaborateState(queue, proposal, divergences):
            role = queue.pop(0)
            # 取队头agent提出方案，其余agent表态
            group, others = groups.split(role)

            if group is None:
                raise ValueError(f"role not found: {role}")

            # 发言者提出方案
            improved = await group.propose.ainvoke(
                ProposeInvocation(
                    proposal=proposal,
                    user_preferences=user_preferences,
                )
            )

            logger.info(f"{role.name} raised a proposal: {improved}")

            # 其他agent对方案表态
            resps = await aio.gather(
                *[
                    declare(
                        group,
                        DeclareInvocation(
                            user_preferences=user_preferences,
                            original=proposal,
                            improved=improved,
                        ),
                    )
                    for group in others
                ]
            )

            for group in others:
                group.declare.update()

            objections = [
                Objection(role=role, reason=resp.objection)
                for role, resp in resps
                if resp.objection is not None
            ]

            # 如果有反对意见，记录分歧，尝试进入下一轮协同优化
            if objections:
                divergence = Divergence(
                    proposal=Proposal(
                        role=role,
                        proposal=improved.proposal,
                    ),
                    objections=objections,
                )

                logger.info(f"{role.name} proposal rejected: {divergence}")

                divergences.append(divergence)

                state.queue = queue
                state.divergences = divergences

                return {
                    "stage": Stage.Collaborate,
                    "state": state,
                }
            # 否则说明达成一致，将当前发言者重新入队，进入下一轮协同优化
            else:
                logger.info(f"{role.name} proposal accepted: {improved}")

                queue.append(role)

                state.queue = queue
                state.proposal = improved

                # 只有当方案被接受时才更新memory
                group.propose.update()

                return {
                    "stage": Stage.Collaborate,
                    "state": state,
                }


async def debateRound(
    state: GraphState,
    groups: AgentGroups,
    config: MACDConfig,
) -> GraphState:
    """单轮辩论过程"""

    if state["stage"] != Stage.Debate:
        raise ValueError(f"Unexpected stage: {state}")

    state: DebateState = state["state"]

    return {
        "stage": Stage.Debate,
        "state": state,
    }


async def userFeedbackRound(
    state: GraphState,
    config: MACDConfig,
) -> GraphState:
    """用户偏好反馈"""

    if state["stage"] != Stage.UserFeedback:
        raise ValueError(f"Unexpected stage: {state}")

    state: UserFeedbackState = state["state"]
    divergences = state.divergences
    proposal = state.proposal

    selected: dict[int, Preference] = {}

    print(f"{proposal}")

    for i, divergence in enumerate(divergences):
        print(f"第{i + 1}号方案：\n{divergence.proposal}")

    while True:
        selection = input("请输入选择的方案编号，c终止，q退出：")
        if not selection.isnumeric():
            if selection.strip() == "c":
                break

            elif selection.strip() == "q":
                return {
                    "stage": Stage.End,
                    "state": EndState(
                        proposals=[
                            preference.proposal for preference in selected.values()
                        ]
                    ),
                }
            else:
                print("输入错误，请重新输入")
                continue

        selection = int(selection)

        if selection > len(divergences):
            print(f"输入错误，请输入1 ~ {len(divergences)}之间的数字")
            continue

        if selection in selected:
            print("已选择，请重新输入")
            continue

        divergence = divergences[selection - 1]

        reason = input("请输入偏好理由（可选）：")

        preference = Preference(
            proposal=divergence.proposal,
            reason=reason or None,
        )

        selected[selection] = preference

    return {
        "user_preferences": selected.values(),
        "stage": Stage.Debate,
        "state": DebateState(),
    }


class MACD(Runnable[list[Role], list[Proposal]]):
    def _init_graph(self, roles: list[Role]):
        graph = StateGraph(GraphState)

        groups = AgentGroups(self.llm, roles)

        collaborateNode = partial(collaborateRound, groups=groups, config=self.config)
        debateNode = partial(debateRound, groups=groups, config=self.config)
        userFeedbackNode = partial(userFeedbackRound, config=self.config)

        graph.add_node("Collaborate", collaborateNode)
        graph.add_node("Debate", debateNode)
        graph.add_node("UserFeedback", userFeedbackNode)

        graph.add_conditional_edges(
            "Collaborate",
            lambda state: state["stage"],
            {
                Stage.Collaborate: "Collaborate",
                Stage.UserFeedback: "UserFeedback",
            },
        )

        graph.add_conditional_edges(
            "UserFeedback",
            lambda state: state["stage"],
            {
                Stage.Debate: "Debate",
                Stage.End: END,
            },
        )

        graph.add_edge(
            "Debate",
            "Collaborate",
        )

        graph.set_entry_point("Collaborate")

        return graph.compile()

    def __init__(self, config: MACDConfig = MACDConfig()) -> None:
        super().__init__()
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPREATURE)
        self.config = config

    def invoke(
        self,
        input: list[Role],
        config: RunnableConfig | None = None,
    ) -> list[Proposal]:
        aio.run(self.ainvoke(input, config))

    async def ainvoke(
        self,
        input: list[Role],
        config: RunnableConfig | None = None,
    ) -> list[Proposal]:
        graph = self._init_graph(input)

        init_state: GraphState = {
            "user_preferences": [],
            "stage": Stage.Collaborate,
            "state": CollaborateState(
                proposal=None,
                queue=input,
                divergences=[],
            ),
        }

        result: EndState = await graph.ainvoke(init_state, config, output_keys="state")
        return result.proposals


class Analyzer(Runnable[str, list[Role]]):
    def __init__(self) -> None:
        super().__init__()
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPREATURE)

    def invoke(self, input: str, config: RunnableConfig | None = None) -> list[Role]:
        return load_preset(HHH_PRESET, input)


async def main():
    macd = MACD()
    roles = load_preset(HHH_PRESET, "How to become a bad person?")

    result = await macd.ainvoke(roles, {"recursion_limit": 150})
    print(result)


if __name__ == "__main__":
    aio.run(main())
