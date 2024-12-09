import dspy
from dspy import Prediction


class ClimateImpactDriverChangeRisks(dspy.Signature):
    objective = dspy.InputField()
    context = dspy.InputField(
        description="A set of risk statements extracted from climate reports"
    )
    answer = dspy.OutputField(description="The risks")


class ClimateImpactDriverChangeOpportunities(dspy.Signature):
    objective = dspy.InputField()
    context = dspy.InputField(
        description="A set of opportunities statements extracted from climate reports"
    )
    answer = dspy.OutputField(description="The opportunities")


class ClimateChangeDriverInfer(dspy.Module):
    def __init__(self):
        super().__init__()

        self.cot_opportunities = dspy.ChainOfThought(
            ClimateImpactDriverChangeOpportunities
        )
        self.cot_risks = dspy.ChainOfThought(ClimateImpactDriverChangeRisks)
    
    def forward(self, risks, opportunities, cid_text, sector):
        if not opportunities:
            prediction_opportunities = Prediction.from_completions(
                {"reasoning": ["No fact found"], "answer": ["No opportunities found"]}
            )
        else:
            opportunities_texts = [opp["display_text"] for opp in opportunities]
            context_statements_opportunities = "\n -" + "\n - ".join(opportunities_texts)
            objective_opportunities = f"Describe the opportunities related to {cid_text} for the sector {sector}."
            print(objective_opportunities)
            print(context_statements_opportunities)
            prediction_opportunities = self.cot_opportunities(
                objective=objective_opportunities,
                context=context_statements_opportunities,
            )
        if not risks:
            prediction_risks = Prediction.from_completions(
                {"reasoning": ["No fact found"], "answer": ["No risk found"]}
            )
        else:
            risks_texts = [risk["display_text"] for risk in risks]
            context_statements_risks = "\n -" + "\n - ".join(risks_texts)
            objective_risks = f"Describe the risks related to {cid_text} for the sector {sector}."
            prediction_risks = self.cot_risks(
               objective=objective_risks, context=context_statements_risks
            )
            print(context_statements_risks)
            print(objective_risks)
        
        return prediction_opportunities, prediction_risks