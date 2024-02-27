from typing import Optional

from pydantic import BaseModel, Field


class Role(BaseModel):
    """
    针对某一议题的角色，包括名称和立场
    """

    topic: str = Field(description="")
    name: str = Field(description="")
    position: str = Field(description="")

    def __str__(self) -> str:
        return f"""{self.name}，立场为：{self.position}"""


class Proposal(BaseModel):
    role: Role = Field(description="")
    proposal: str = Field(description="")

    def __str__(self) -> str:
        return f"""{self.role}，提出方案：
        
        {self.proposal}"""


class Objection(BaseModel):
    role: Role = Field(description="")
    reason: str = Field(description="")

    def __str__(self) -> str:
        return f"""{self.role}，提出了反对，反对理由：
        
        {self.reason}"""


class Divergence(BaseModel):
    """
    协同优化过程中，某一提案及其反对意见
    """

    proposal: Proposal = Field(description="")
    objections: list[Objection] = Field(description="")

    def __str__(self) -> str:
        objections = "\n".join(str(o) for o in self.objections)

        return f"""{self.proposal}

        反对意见如下：

        {objections}
        """


class Preference(BaseModel):
    proposal: Proposal = Field(description="The proposal")
    reason: Optional[str] = Field(description="The reason for the preference")

    def __str__(self) -> str:
        return f"""用户偏好如下方案：
        
        {self.proposal}
        
        原因是：
        
        {self.reason}
        """


class Preferences(list[Preference]):
    def __str__(self) -> str:
        return "\n".join(str(p) for p in self)
