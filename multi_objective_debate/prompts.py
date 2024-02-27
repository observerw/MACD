from typing import Optional, TypedDict

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from multi_objective_debate.model import Preferences, Proposal

# ----------------------------------- 系统提示 ----------------------------------- #

system_template = SystemMessagePromptTemplate.from_template(
    template="""你是一名顾问，需要与其他持有不同立场的顾问进行合作和辩论，从而更好的解决用户的需求。
    
    用户的需求为：
    
    {topic}

    你的名字是：
    
    {name}
    
    如下是你对于该需求所秉持的立场：
    
    {position}
    
    请你相信自己立场的正当性，在任何时候都不要完全接受其他守护者的立场。
    """
)

# ------------------------------------ 共有 ------------------------------------ #


class Invocation(TypedDict):
    user_preferences: Preferences


preference_template = HumanMessagePromptTemplate.from_template(
    """用户目前更为偏好的方案和偏好理由：

    {user_preferences}"""
)

format_template = HumanMessagePromptTemplate.from_template("{format_instuctions}")

# ------------------------------------- 竞争优化和协同优化-提出方案 ------------------------------------ #


class ProposeInvocation(Invocation):
    proposal: Proposal


class ProposeResponse(BaseModel):
    proposal: str = Field(description="The proposal")


propose_parser = PydanticOutputParser(pydantic_object=ProposeResponse)

propose_memory_template = HumanMessagePromptTemplate.from_template(
    """目前的最优方案：

    {proposal}

    改进方案：
    """,
)

propose_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        """请你参考改良经验和用户的偏好，尝试从你的立场出发提出一个更为符合用户需求的方案。

        请你一步步的进行思考，然后给出你的答复，思考过程应该尽量客观简洁。
        """,
        MessagesPlaceholder(variable_name="memory"),
        preference_template,
        format_template,
        propose_memory_template,
    ],
).partial(format_instuctions=propose_parser.get_format_instructions())


# ---------------------------------- 协同优化-表态 --------------------------------- #


class DeclareInvocation(Invocation):
    original: Proposal
    improved: Proposal


class DeclareResponse(BaseModel):
    objection: Optional[str] = Field(
        description="Whether to accept the improved proposal, if not, provide the reason for the objection"
    )


declare_parser = PydanticOutputParser(pydantic_object=DeclareResponse)

declare_memory_template = HumanMessagePromptTemplate.from_template(
    """原方案：

    {original}

    改进方案：

    {improved}

    表态：
    """
)

declare_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        """如下是用户目前更为偏好的方案和偏好理由：

        {user_preferences}

        请你参考过去的表决经验和用户的偏好，仔细思考：该方案从你的角度而言是否可以接受？可接受的标准为：
    
        - 该方案不会损害到当前立场的利益，甚至可能会对当前立场的利益产生积极影响；
        - 该方案可能会对当前立场的利益产生一定限度的负面影响，但这一影响是可以通过后续改进而弥补的，并非无可挽回的损失；

        请你一步步的进行思考，然后给出你的答复，思考过程应该尽量客观简洁。
        """,
        MessagesPlaceholder(variable_name="memory"),
        preference_template,
        format_template,
        declare_memory_template,
    ]
).partial(format_instuctions=declare_parser.get_format_instructions())

# --------------------------------- 竞争优化-顺序改进 -------------------------------- #


class ImproveInvocation(Invocation):
    pass


class ImproveResponse(BaseModel):
    proposal: str = Field(description="The proposal")


improve_parser = PydanticOutputParser(pydantic_object=ImproveResponse)

improve_memory_template = HumanMessagePromptTemplate.from_template("""""")

improve_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        """
        请你参考过去的改良经验和用户的偏好，尝试从你的立场出发提出一个更为符合用户需求的方案。

        请你一步步的进行思考，然后给出你的答复，思考过程应该尽量客观简洁。

        {format_instuctions}
        """,
        MessagesPlaceholder(variable_name="memory"),
        preference_template,
        improve_memory_template,
    ],
).partial(format_instuctions=improve_parser.get_format_instructions())

# ---------------------------------- 竞争优化-作家 --------------------------------- #


class WriteInvocation(Invocation):
    pass


class WriteResponse(BaseModel):
    proposal: str = Field(description="The proposal")


write_parser = PydanticOutputParser(pydantic_object=WriteResponse)

write_memory_template = HumanMessagePromptTemplate.from_template(
    """原方案：

    {proposal}

    如下是其他顾问对于该方案提出的批评意见：

    {criticisms}

    改进方案：
    """
)

write_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        """请你参考过去的改良经验和所有批评意见，尝试提出一个能够解决批评意见的方案，同时也需要考虑到自身的立场。

        请你一步步的进行思考，然后给出你的答复，思考过程应该尽量客观简洁。

        {format_instuctions}
        """,
        MessagesPlaceholder(variable_name="memory"),
        preference_template,
        write_memory_template,
    ],
).partial(format_instuctions=write_parser.get_format_instructions())

# --------------------------------- 竞争优化-批评家 --------------------------------- #


class CriticizeInvocation(Invocation):
    pass


class CriticizeResponse(BaseModel):
    proposal: str = Field(description="The proposal")


criticize_parser = PydanticOutputParser(pydantic_object=CriticizeResponse)

critize_memory_template = HumanMessagePromptTemplate.from_template(
    """
    当前方案：

    {proposal}

    批评意见：
    """
)

criticize_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        """请你参考过去的改良经验和用户的偏好，尝试从你的立场出发提出一个更为符合用户需求的方案。

        请你一步步的进行思考，然后给出你的答复，思考过程应该尽量客观简洁。

        {format_instuctions}
        """,
        MessagesPlaceholder(variable_name="memory"),
        preference_template,
        critize_memory_template,
    ],
).partial(format_instuctions=criticize_parser.get_format_instructions())
