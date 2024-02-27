import logging
from typing import Optional

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

import multi_objective_debate.prompts as prompts
from multi_objective_debate.model import Role


class Agent[Input, Output](Runnable[Input, Output]):
    def __init__(
        self,
        llm: ChatOpenAI,
        system_message: SystemMessage,
        template: ChatPromptTemplate,
        parser: PydanticOutputParser,
    ) -> None:
        self.system_message = system_message
        self.memory: list[BaseMessage] = []

        self.llm = llm
        self.template = template
        self.parser = parser

        self.temp: Optional[tuple[HumanMessage, AIMessage]] = None

    def invoke(self, input: Input, config: RunnableConfig | None = None) -> Output:
        messages = self.template.format_messages(
            system=[self.system_message],
            memory=self.memory,
            **input,
        )

        logging.info(f"invoke with messages: {messages}")

        resp = self.llm.invoke(messages, config)

        # 最后一个message代表本次invoke的输入，该message与agent的response一起存入memory
        self.temp = (messages[-1], resp)

        return self.parser.invoke(resp, config)

    def update(self):
        """
        更新memory

        默认行为是将本次invoke的输入和响应存入memory

        不同子agent可能具有不同的memory更新策略
        """
        if self.temp is None:
            raise ValueError("please invoke first before update")

        self.memory.extend(self.temp)

        self.temp = None


class ProposeAgent(Agent[prompts.ProposeInvocation, prompts.ProposeResponse]):
    pass


class DeclareAgent(Agent[prompts.DeclareInvocation, prompts.DeclareResponse]):
    pass


class ImproveAgent(Agent[prompts.ImproveInvocation, prompts.ImproveResponse]):
    pass


class WriteAgent(Agent[prompts.WriteInvocation, prompts.WriteResponse]):
    pass


class CriticizeAgent(Agent[prompts.CriticizeInvocation, prompts.CriticizeResponse]):
    pass


class AgentGroup:
    """
    同一个组里的agent采用相同的system message
    """

    def __init__(self, llm: ChatOpenAI, role: Role) -> None:
        self.role = role

        system_message = prompts.system_template.format(**role.model_dump())

        self.propose = ProposeAgent(
            llm,
            system_message,
            prompts.propose_template,
            prompts.propose_parser,
        )

        self.declare = DeclareAgent(
            llm,
            system_message,
            prompts.declare_template,
            prompts.declare_parser,
        )

        self.improve = ImproveAgent(
            llm,
            system_message,
            prompts.improve_template,
            prompts.improve_parser,
        )

        self.write = WriteAgent(
            llm,
            system_message,
            prompts.write_template,
            prompts.write_parser,
        )

        self.criticize = CriticizeAgent(
            llm,
            system_message,
            prompts.criticize_template,
            prompts.criticize_parser,
        )


class AgentGroups:
    mapping: dict[str, AgentGroup]

    def __init__(self, llm: ChatOpenAI, roles: list[Role]) -> None:
        self.roles = roles
        self.mapping = {role.name: AgentGroup(llm, role) for role in roles}

    def get(self, role: Role) -> Optional[AgentGroup]:
        return self.mapping.get(role.name)

    def roles(self) -> list[Role]:
        return self.roles

    def split(self, role: Role) -> tuple[Optional[AgentGroup], list[AgentGroup]]:
        """
        返回指定role的group和其他group
        """
        group = self.get(role)
        others = [v for k, v in self.mapping.items() if k != role.name]

        return group, others
