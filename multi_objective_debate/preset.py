from multi_objective_debate.model import Role


HHH_PRESET = {
    "name": "HHH",
    "roles": [
        {
            "name": "Helpful",
            "position": "The answer should provide enough information to help the user solve the problem.",
        },
        {
            "name": "Honest",
            "position": "The answer should be honest and not deceive the user.",
        },
        {
            "name": "Harmless",
            "position": "The answer should not cause harm to the user.",
        },
    ],
}


def load_preset(preset: dict, topic: str) -> list[Role]:
    return [Role(topic=topic, **role) for role in preset["roles"]]
