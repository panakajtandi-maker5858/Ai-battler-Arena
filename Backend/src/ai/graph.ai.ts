import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { mistralAIModel, cohereModel, groqModel } from "./model.ai.js";

// State
const GraphState = Annotation.Root({
    problem: Annotation<string>({
        reducer: (_x, y) => y,
        default: () => "",
    }),
    solution_1: Annotation<string>({
        reducer: (_x, y) => y,
        default: () => "",
    }),
    solution_2: Annotation<string>({
        reducer: (_x, y) => y,
        default: () => "",
    }),
    judge: Annotation<{
        solution_1_score: number;
        solution_2_score: number;
        solution_1_reasoning: string;
        solution_2_reasoning: string;
    }>({
        reducer: (_x, y) => y,
        default: () => ({
            solution_1_score: 0,
            solution_2_score: 0,
            solution_1_reasoning: "",
            solution_2_reasoning: "",
        }),
    }),
});

// Judge ka schema
const judgeSchema = z.object({
    solution_1_score: z.number().min(0).max(10),
    solution_2_score: z.number().min(0).max(10),
    solution_1_reasoning: z.string(),
    solution_2_reasoning: z.string(),
});

// Solution Node
const solutionNode = async (state: typeof GraphState.State) => {
    const [mistralResponse, cohereResponse] = await Promise.all([
        mistralAIModel.invoke(state.problem),
        cohereModel.invoke(state.problem),
    ]);

    return {
        solution_1: mistralResponse.content as string,
        solution_2: cohereResponse.content as string,
    };
};

// Judge Node
const judgeNode = async (state: typeof GraphState.State) => {
    const { problem, solution_1, solution_2 } = state;

    const judgeWithStructuredOutput = groqModel.withStructuredOutput(judgeSchema);

    const judgeResponse = await judgeWithStructuredOutput.invoke([
        new HumanMessage(`
            You are a judge evaluating two AI solutions.
            Problem: ${problem}
            Solution 1: ${solution_1}
            Solution 2: ${solution_2}
            Score each solution out of 10 and give reasoning.
        `),
    ]);

    return {
        judge: {
            solution_1_score: judgeResponse.solution_1_score,
            solution_2_score: judgeResponse.solution_2_score,
            solution_1_reasoning: judgeResponse.solution_1_reasoning,
            solution_2_reasoning: judgeResponse.solution_2_reasoning,
        },
    };
};

// Graph
const graph = new StateGraph(GraphState)
    .addNode("solution", solutionNode)
    .addNode("judge_node", judgeNode)
    .addEdge(START, "solution")
    .addEdge("solution", "judge_node")
    .addEdge("judge_node", END)
    .compile();

export default async function runGraph(problem: string) {
    const result = await graph.invoke({ problem });
    return result;
}