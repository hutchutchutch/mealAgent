import { ChatOpenAI, OpenAI } from "@langchain/openai";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { z } from "zod";
import { RunnableConfig } from "@langchain/core/runnables";

// Schema Definitions
const RecipeSchema = z.object({
    name: z.string(),
    description: z.string(),
    tips: z.string(),
    calories: z.number(),
    protein: z.number(),
    ingredients: z.array(z.object({
        name: z.string(),
        quantity: z.string(),
        unit: z.string(),
        calories: z.number(),
        protein: z.number(),
        flavorProfile: z.string()
    })),
    cookTime: z.string(),
    instructions: z.array(z.string()),
    dietaryTags: z.array(z.string()),
    allergens: z.array(z.string())
});

const ValidationResultSchema = z.array(z.object({
    rule: z.string(),
    status: z.enum(["pass", "fail"])
}));

const ShoppingListSchema = z.record(z.array(z.object({
    item: z.string(),
    quantity: z.string(),
    unit: z.string()
})));

const MealPlanningStateAnnotation = Annotation.Root({
    // I don't think we need this BaseMessage in the state
    //    messages: Annotation<BaseMessage[]>(),
    userPreferences: Annotation<{
        caloriesPerMeal: number;
        proteinPerMeal: number;
        allergens: string[];
        dietaryPreferences: string[];
        likedFoods: string[];
        dislikedFoods: string[];
        location: string; // for seasonal foods
    }>(),
    recipes: Annotation<z.infer<typeof RecipeSchema>[]>(),
    validationRules: Annotation<z.infer<typeof ValidationResultSchema>[]>(),
    shoppingList: Annotation<z.infer<typeof ShoppingListSchema>[]>()
});


// Initialize LLM
const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini"
});

const recipeGenerator = async (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> => {
    const prompt = `{
  "role": {
    "identity": "You are both a warm, encouraging home cooking expert and a trained nutritionist who deeply understands both the culinary arts and the science of nutrition. You have an encyclopedic knowledge of the nutritional content of ingredients - from the protein content in different cuts of meat to the precise caloric value of oils and vegetables. You understand not just how ingredients work together for flavor, but also how they combine to create nutritionally balanced meals. Your expertise lies in helping home cooks create delicious dishes while teaching them about the nutritional value of each ingredient they use. In each step of your instructions, you give useful tips for the home cook to understand what they are looking for and what they should avoid."
  },
  "principles": {
    "salt": "Learning to season confidently with simple combinations of spices",
    "fat": "Using healthy fats strategically for both flavor and nutritional benefits",
    "acid": "Adding brightness while preserving nutrients in fresh ingredients",
    "nutrition": "Understanding the precise nutritional contribution of each ingredient",
    "balance": "Creating meals that are simple, delicious, balanced in texture and flavor, and nutritionally sound"
  },
  "recipeParameters": {
    "targetCalories": "${state.userPreferences.caloriesPerMeal} calories",
    "targetProtein": "${state.userPreferences.proteinPerMeal}g",
    "allergensToAvoid": "${state.userPreferences.allergens.join(', ')}",
    "dietaryGuidelines": "${state.userPreferences.dietaryPreferences.join(', ')}",
    "seasonality": {
      "month": "${new Date().toLocaleString('default', { month: 'long' })}",
      "region": "${state.userPreferences.location}"
    },
    "ingredientsToAvoid": "${state.userPreferences.dislikedFoods.join(', ')}",
    "maxCookingTime": "20 minutes"
  },
  "task": "Create a recipe that makes seasonal cooking feel effortless while teaching basic principles of flavor and texture. Focus on techniques that are approachable and build confidence in the kitchen.",
  "examples": [
    {
      "name": "Crispy Pan-Seared Chicken with Lemony Arugula Salad",
      "description": "This simple but satisfying dish shows how a few basic techniques can create amazing textures and flavors. The chicken develops a golden-brown crust while staying juicy inside - a perfect contrast to the fresh, peppery arugula. The lemony dressing brings everything together, while the natural fat from the chicken adds richness. Every bite offers a delightful mix of warm and cool, crispy and tender, bright and savory that makes this easy dish feel special.",
      "tips": "Don't be afraid of the sizzle when you add the chicken to the pan - that sound means you're getting good browning! Pat the chicken dry with paper towels before cooking; this helps create that golden crust. If the arugula seems too peppery, you can mix in some soft lettuce. Make extra dressing and keep it in a jar for quick salads throughout the week. Let the chicken rest for a few minutes after cooking so the juices stay in the meat when you cut it.",
      "calories": 410,
      "protein": 35,
      "ingredients": [
        {
          "name": "chicken breast",
          "quantity": "6",
          "unit": "oz",
          "calories": 180,
          "protein": 28,
          "flavorProfile": "mild, savory, lean protein that takes on flavors well"
        },
        {
          "name": "baby arugula",
          "quantity": "3",
          "unit": "cups",
          "calories": 15,
          "protein": 2,
          "flavorProfile": "peppery, fresh, slightly bitter greens with a crisp texture"
        },
        {
          "name": "lemon",
          "quantity": "1",
          "unit": "whole",
          "calories": 12,
          "protein": 0,
          "flavorProfile": "bright, acidic, citrusy with floral notes"
        },
        {
          "name": "olive oil",
          "quantity": "3",
          "unit": "tbsp",
          "calories": 120,
          "protein": 0,
          "flavorProfile": "rich, fruity, with a peppery finish"
        },
        {
          "name": "kosher salt",
          "quantity": "1",
          "unit": "tsp",
          "calories": 0,
          "protein": 0,
          "flavorProfile": "clean, mineral taste that enhances other flavors"
        },
        {
          "name": "black pepper",
          "quantity": "1/4",
          "unit": "tsp",
          "calories": 0,
          "protein": 0,
          "flavorProfile": "sharp, spicy, aromatic"
        }
      ],
      "cookTime": "15 minutes",
      "instructions": [
"Pat chicken dry with paper towels and season both sides with salt and pepper. Make sure to really pat the chicken completely dry - any excess moisture will prevent proper browning and can cause dangerous oil splattering. Don't be shy with the seasoning, as some will fall off during cooking.",
"Heat 2 tablespoons oil in a pan over medium-high heat until it shimmers. Watch carefully for that shimmering effect - if you see smoke, the pan is too hot and the oil may be burning. The oil should move like water when the pan is tilted.",
"Add chicken and cook without moving for 5-6 minutes until golden brown. Resist the urge to move the chicken! Letting it sit undisturbed creates that golden crust. If the chicken sticks when you try to flip it, it likely needs more time to brown.",
"Flip and cook 4-5 minutes more until cooked through. The best way to check doneness is with a meat thermometer - it should read 165°F (74°C) at the thickest part. The meat should feel firm but not hard when pressed.",
"While chicken cooks, squeeze half the lemon into a bowl with remaining oil. Squeeze the lemon through your cupped fingers to catch any seeds. Roll the lemon on the counter first to get more juice.",
"Add a pinch of salt and pepper to the dressing and whisk with a fork. Taste as you season - you can always add more salt but can't take it out. The dressing should be slightly more intense than you think, as it will be spread over the entire dish.",
"Toss arugula with half the dressing. Use clean hands to gently toss - tongs can bruise the delicate leaves. Add dressing gradually to avoid overdressing and wilting the greens.",
"Slice chicken, arrange over arugula, and drizzle with remaining dressing. Let the chicken rest for 5 minutes before slicing to keep the juices in. Slice against the grain for maximum tenderness, and serve immediately while the chicken is still warm and the arugula crisp."

      ],
      "dietaryTags": [
        "gluten-free",
        "dairy-free",
        "low-carb"
      ],
      "allergens": [
        "none"
      ]
    },
    {
      "name": "Summer Tomato and White Bean Toast",
      "description": "This no-cook meal celebrates the perfect summer tomato while teaching us about balance. The creamy beans provide a rich base that's brightened by sweet, juicy tomatoes. Torn basil adds freshness, while a drizzle of olive oil brings luxurious richness. A final sprinkle of salt makes all the flavors pop. The contrast between the crispy toast and creamy toppings makes every bite interesting, proving that simple ingredients can create amazing meals.",
      "tips": "The type of tomato matters - use the ripest ones you can find and save the hard winter tomatoes for cooking. Toast the bread until it's really golden; this creates a sturdy base that won't get soggy. If you have time, let your beans come to room temperature for the best flavor. Don't skip the final drizzle of olive oil and sprinkle of salt - these finishing touches make a big difference. Rub a peeled garlic clove on the hot toast for extra flavor.",
      "calories": 380,
      "protein": 22,
      "ingredients": [
        {
          "name": "crusty bread",
          "quantity": "2",
          "unit": "slices",
          "calories": 160,
          "protein": 6,
          "flavorProfile": "nutty, toasted wheat with crispy exterior and chewy interior"
        },
        {
          "name": "canned white beans",
          "quantity": "15",
          "unit": "oz",
          "calories": 150,
          "protein": 14,
          "flavorProfile": "mild, creamy, slightly earthy with buttery texture"
        },
        {
          "name": "ripe tomatoes",
          "quantity": "2",
          "unit": "medium",
          "calories": 35,
          "protein": 1,
          "flavorProfile": "sweet, bright, slightly acidic with umami notes"
        },
        {
          "name": "fresh basil",
          "quantity": "1/4",
          "unit": "cup",
          "calories": 5,
          "protein": 0,
          "flavorProfile": "aromatic, peppery, slightly sweet with anise notes"
        },
        {
          "name": "olive oil",
          "quantity": "2",
          "unit": "tbsp",
          "calories": 240,
          "protein": 0,
          "flavorProfile": "rich, fruity, with a peppery finish"
        },
        {
          "name": "kosher salt",
          "quantity": "1/2",
          "unit": "tsp",
          "calories": 0,
          "protein": 0,
          "flavorProfile": "clean, mineral taste that enhances other flavors"
        }
      ],
      "cookTime": "10 minutes",
      "instructions": [
        "Drain and rinse beans, then mash roughly with a fork",
        "Season beans with half the salt and 1 tablespoon olive oil",
        "Toast bread until golden brown",
        "Slice tomatoes",
        "Spread mashed beans on toast",
        "Top with tomato slices",
        "Tear basil leaves over top",
        "Finish with remaining olive oil and salt"
      ],
      "dietaryTags": [
        "vegetarian",
        "dairy-free"
      ],
      "allergens": [
        "wheat"
      ]
    }
  ],
  "requirements": {
    "rules": [
      "Contain 6 main ingredients or less (excluding oils, salt, pepper, and basic seasonings)",
      "Meet all specified nutritional requirements within a 5% margin",
      "Use simple techniques that build kitchen confidence",
      "Create exciting flavors through basic seasoning",
      "Focus on readily available seasonal ingredients",
      "Require only basic kitchen equipment",
      "Be completable within 20 minutes",
      "Include everyday measurements that home cooks understand",
      "Avoid all listed allergens and ingredients",
      "Provide clear, encouraging instructions",
      "Include practical tips that help develop cooking intuition",
      "Detail the nutritional contribution of each ingredient",
      "Describe the flavor profile of each ingredient"
    ],
    "ingredientDetails": {
      "required": [
        "Accurate caloric content",
        "Protein content",
        "A detailed flavor profile that helps home cooks understand its contribution to the dish"
      ]
    },
    "description": "In your description, explain how simple ingredients work together to create delicious results. Highlight how different textures make the dish interesting. Your tips should focus on building confidence and understanding why certain steps matter.",
    "style": "Use clear, friendly language that encourages home cooks to try new techniques."
  }
}
`;

    const response = await model.invoke([new HumanMessage(prompt)]);

    // Parse and validate the response
    const recipe = RecipeSchema.parse(JSON.parse(response.content as string));

    return { recipe };
};


// Recipe Validator Node
const recipeValidator = async (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> => {
    const prompt = `{
  "role": {
    "title": "Expert Food Scientist and Recipe Validator",
    "expertise": [
      "Nutritional composition of ingredients",
      "Food allergens and their derivatives",
      "Dietary restrictions and guidelines",
      "Recipe timing and preparation methods",
      "Kitchen techniques and cooking times"
    ]
  },
  "task": {
    "description": "Analyze recipe and validate against requirements",
    "inputs": {
      "recipe": ${JSON.stringify(state.recipe, null, 2)},
      "nutritionalTargets": {
        "calories": ${state.userPreferences.caloriesPerMeal},
        "protein": ${state.userPreferences.proteinPerMeal}
      },
      "allergens": ${JSON.stringify(state.userPreferences.allergens)},
      "dietaryGuidelines": ${JSON.stringify(state.userPreferences.dietaryPreferences)},
      "restrictedIngredients": ${JSON.stringify(state.userPreferences.dislikedFoods)}
    }
  },
  "validationRules": {
    "ingredientCountRule": {
      "description": "Recipe must contain 6 or fewer main ingredients",
      "excludes": ["oils", "salt", "pepper", "basic seasonings"],
      "maxCount": 6
    },
    "calorieComplianceRule": {
      "description": "Total calories must be within ±15% of target",
      "tolerance": 0.15
    },
    "proteinComplianceRule": {
      "description": "Total protein must be within ±15% of target",
      "tolerance": 0.15
    },
    "allergenSafetyRule": {
      "description": "No listed allergens can be present in any form"
    },
    "dietaryComplianceRule": {
      "description": "Recipe must follow all specified dietary guidelines"
    },
    "ingredientRestrictionRule": {
      "description": "No disliked/restricted ingredients can be present"
    },
    "timeManagementRule": {
      "description": "Recipe must be completable in under 20 minutes",
      "maxMinutes": 20
    }
  },
  "requiredOutputFormat": [
    {
      "rule": "ruleName",
      "status": "pass or fail"
    }
  ],
  "note": "Return an array of validation results for all rules, following the output format exactly. Each rule should have a status of either 'pass' or 'fail'. Return only the JSON array with no additional explanation."
}`;

    const response = await model.invoke([new HumanMessage(prompt)]);
    
    // Parse and validate the response
    const validationResults = ValidationResultSchema.parse(JSON.parse(response.content as string));

    return { validationResults };
};


// Recipe Editor Node
const recipeEditor = async (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> => {
    const failedRules = state.validationResults?.filter(rule => rule.status === "fail");

    const prompt = `Edit this recipe to fix the following issues:
  Recipe: ${JSON.stringify(state.recipe, null, 2)}
  Failed validations: ${JSON.stringify(failedRules, null, 2)}
  
  User Requirements:
  - Target calories: ${state.userPreferences.caloriesPerMeal}
  - Target protein: ${state.userPreferences.proteinPerMeal}
  - Allergens to avoid: ${state.userPreferences.allergens.join(', ')}
  - Dietary preferences: ${state.userPreferences.dietaryPreferences.join(', ')}
  - Disliked foods: ${state.userPreferences.dislikedFoods.join(', ')}
  
  Provide the complete edited recipe in the same structured format.`;

    const response = await model.invoke([new HumanMessage(prompt)]);

    // Parse and validate the response
    const recipe = RecipeSchema.parse(JSON.parse(response.content as string));

    return { recipe };
};

// Shopping List Generator Node
const shoppingListGenerator = async (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> => {
    const prompt = `Create a categorized shopping list from this recipe:
  ${JSON.stringify(state.recipe, null, 2)}
  
  Group items by category (produce, dairy, pantry, etc.) and include quantities.
  Return as a JSON object where each category contains an array of items with quantities.`;

    const response = await model.invoke([new HumanMessage(prompt)]);

    // Parse and validate the response
    const shoppingList = ShoppingListSchema.parse(JSON.parse(response.content as string));

    return { shoppingList };
};

// Validation routing logic
const shouldContinueValidation = (state: typeof MealPlanningStateAnnotation.State) => {
    // Check if validationResults exists and if any rule has "fail" status
    const hasFailedRules = state.validationResults?.some(rule => rule.status === "fail");
    
    // If there are failed rules, return "recipe_editor", otherwise return END
    return hasFailedRules ? "recipe_editor" : END;
};

// Create main graph
const createMainGraph = () => {
    // const workflow = new StateGraph<RecipeState>();
    const workflow = new StateGraph({
        stateSchema: MealPlanningStateAnnotation
    });

    // Add nodes
    workflow.addNode("generator", recipeGenerator);
    workflow.addNode("validator", recipeValidator);
    workflow.addNode("editor", recipeEditor);
    workflow.addNode("shopping_list", shoppingListGenerator);

    // Add edges
    workflow.addEdge(START, "generator");
    workflow.addEdge("generator", "validator");
    workflow.addConditionalEdges(
        "validator",
        shouldContinueValidation,
        {
            "recipe_editor": "editor",
            [END]: "shopping_list"
        }
    );
    workflow.addEdge("validator", "shopping_list");
    workflow.addEdge("shopping_list", END);

    return workflow.compile({ checkpointer: new MemorySaver() });

};
// Create the application
const app = createMainGraph();

// Example usage
const runWorkflow = async () => {
    const initialState: typeof MealPlanningStateAnnotation.State = {
        userPreferences: {
            caloriesPerMeal: 500,
            proteinPerMeal: 20,
            allergens: ["peanuts", "tree nuts"],
            dietaryPreferences: ["vegetarian"],
            dislikedFoods: ["garlic", "onions"],
            location: "California"
        },
        recipe: undefined,
        validationResults: undefined,
        shoppingList: undefined
    };

    const finalState = await app.invoke(initialState);
    return finalState;
};


export { app, runWorkflow, MealPlanningStateAnnotation, RecipeSchema, ValidationResultSchema, ShoppingListSchema };