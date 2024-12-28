import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { z } from "zod";
import { RunnableConfig } from "@langchain/core/runnables";
import { StructuredOutputParser } from "@langchain/core/output_parsers";

// Schema Definitions
export const RecipeSchema = z.object({
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

export const ValidationResultSchema = z.array(z.object({
    rule: z.string(),
    status: z.enum(["pass", "fail"])
}));

export const ShoppingListSchema = z.record(z.array(z.object({
    item: z.string(),
    quantity: z.string(),
    unit: z.string()
})));

export const MealPlanningStateAnnotation = Annotation.Root({
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


// // Initialize LLM
// const model = new ChatOpenAI({
//     openAIApiKey: process.env.OPENAI_API_KEY,
//     modelName: "gpt-4o-mini"
// });

async function recipeGenerator (
    state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> {
    try {
        
        console.log("state in recipeGenerator ", state);
        // Initialize LLM
        const model = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: "gpt-4o-mini"
        });
    
        const parser = StructuredOutputParser.fromZodSchema(RecipeSchema.array());
    
    
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
        "targetCalories": "${state.userPreferences?.caloriesPerMeal} calories",
        "targetProtein": "${state.userPreferences?.proteinPerMeal}g",
        "allergensToAvoid": "${state.userPreferences?.allergens.join(', ')}",
        "dietaryGuidelines": "${state.userPreferences?.dietaryPreferences.join(', ')}",
        "seasonality": {
          "month": "${new Date().toLocaleString('default', { month: 'long' })}",
          "region": "${state.userPreferences?.location}"
        },
        "ingredientsToAvoid": "${state.userPreferences?.dislikedFoods.join(', ')}",
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
    
    
    **FORMATED INSTRUCTIONS**
      ${parser.getFormatInstructions()}
    
        `;
    
        const response = await model.invoke([new HumanMessage(prompt)]);
        const content = response?.content;
        console.log("content in recipeGenerator ", content);
    
    
        try {
            let cleanedContent = '';
    
            if (typeof response.content === 'string') {
                cleanedContent = response.content
                    .replace(/```(?:json)?|```/g, "")
                    .trim();
            } else if (Array.isArray(response.content)) {
                cleanedContent = JSON.stringify(response.content);
            } else {
                cleanedContent = JSON.stringify(response.content);
            }
    
            const recipes = JSON.parse(cleanedContent);

            console.log('recipes in recipeGenerator', recipes )
    
            return { recipes };
        } catch (error) {
            console.error("Failed to generate recipes:", error);
            return { recipes: [] };
        }
    } catch (error) {
        console.error("Failed to generate recipes:", error);
        return { recipes: [] };
    }

};


// Recipe Validator Node
async function recipeValidator (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update>{
    try {
        console.log("state in recipeValidator", state)
        // Initialize LLM
        const model = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: "gpt-4o-mini"
        });
    
        const parser = StructuredOutputParser.fromZodSchema(ValidationResultSchema.array());
    
    
        const prompt = `
        {
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
          "description": "Analyze recipes and validate them against specific requirements",
          "inputs": [
            "Complete recipe with ingredients, instructions, and nutritional information",
            "targetCalories": "${state.userPreferences.caloriesPerMeal} calories",
            "targetProtein": "${state.userPreferences.proteinPerMeal}g",
            "Allergen restrictions: ${state.userPreferences.allergens.join(', ')}",
            "Dietary guidelines: ${state.userPreferences.dietaryPreferences.join(', ')}",
            "Ingredients to avoid: [${state.userPreferences.dislikedFoods.join(', ')}]",
          ]
        },
        "validationRules": {
          "ingredientCountRule": {
            "description": "Recipe must contain 6 or fewer main ingredients",
            "excludes": [
              "oils",
              "salt",
              "pepper",
              "basic seasonings"
            ],
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
        "examples": {
          "passingExample": {
            "input": {
              "recipe": {
                "name": "Pan-Seared Salmon with Green Beans",
                "ingredients": [
                  {
                    "name": "salmon fillet",
                    "quantity": "6",
                    "unit": "oz",
                    "calories": 240,
                    "protein": 34,
                    "flavorProfile": "rich, fatty fish"
                  },
                  {
                    "name": "green beans",
                    "quantity": "2",
                    "unit": "cups",
                    "calories": 70,
                    "protein": 4,
                    "flavorProfile": "crisp, fresh"
                  },
                  {
                    "name": "lemon",
                    "quantity": "1",
                    "unit": "whole",
                    "calories": 12,
                    "protein": 0,
                    "flavorProfile": "bright, acidic"
                  }
                ],
                "cookTime": "15 minutes",
                "instructions": [
                  "Pat salmon dry...",
                  "Steam green beans..."
                ],
                "allergens": [
                  "fish"
                ]
              },
              "targetValues": {
                "calories": 300,
                "protein": 35,
                "allergensToAvoid": [
                  "dairy",
                  "nuts",
                  "soy"
                ],
                "dietaryGuidelines": [
                  "low-carb",
                  "gluten-free"
                ],
                "ingredientsToAvoid": [
                  "mushrooms",
                  "onions"
                ]
              }
            },
            "output": {
              "ingredientCountRule": "pass",
              "calorieComplianceRule": "pass",
              "proteinComplianceRule": "pass",
              "allergenSafetyRule": "pass",
              "dietaryComplianceRule": "pass",
              "ingredientRestrictionRule": "pass",
              "timeManagementRule": "pass"
            }
          },
          "failingExample": {
            "input": {
              "recipe": {
                "name": "Creamy Mushroom Pasta",
                "ingredients": [
                  {
                    "name": "fettuccine pasta",
                    "quantity": "8",
                    "unit": "oz",
                    "calories": 400,
                    "protein": 14,
                    "flavorProfile": "neutral, wheaty"
                  },
                  {
                    "name": "heavy cream",
                    "quantity": "1",
                    "unit": "cup",
                    "calories": 821,
                    "protein": 5,
                    "flavorProfile": "rich, creamy"
                  },
                  {
                    "name": "mushrooms",
                    "quantity": "8",
                    "unit": "oz",
                    "calories": 50,
                    "protein": 7,
                    "flavorProfile": "earthy, umami"
                  },
                  {
                    "name": "parmesan cheese",
                    "quantity": "1/2",
                    "unit": "cup",
                    "calories": 215,
                    "protein": 19,
                    "flavorProfile": "salty, umami"
                  },
                  {
                    "name": "garlic",
                    "quantity": "4",
                    "unit": "cloves",
                    "calories": 16,
                    "protein": 0,
                    "flavorProfile": "pungent, aromatic"
                  },
                  {
                    "name": "shallots",
                    "quantity": "2",
                    "whole": "whole",
                    "calories": 72,
                    "protein": 2,
                    "flavorProfile": "mild onion"
                  },
                  {
                    "name": "fresh thyme",
                    "quantity": "4",
                    "unit": "sprigs",
                    "calories": 4,
                    "protein": 0,
                    "flavorProfile": "herbal, earthy"
                  }
                ],
                "cookTime": "25 minutes",
                "instructions": [
                  "Boil pasta...",
                  "Sauté mushrooms..."
                ],
                "allergens": [
                  "wheat",
                  "dairy"
                ]
              },
              "targetValues": {
                "calories": 800,
                "protein": 35,
                "allergensToAvoid": [
                  "dairy",
                  "wheat"
                ],
                "dietaryGuidelines": [
                  "low-carb"
                ],
                "ingredientsToAvoid": [
                  "mushrooms"
                ]
              }
            },
            "output": {
              "ingredientCountRule": "fail",
              "calorieComplianceRule": "fail",
              "proteinComplianceRule": "pass",
              "allergenSafetyRule": "fail",
              "dietaryComplianceRule": "fail",
              "ingredientRestrictionRule": "fail",
              "timeManagementRule": "fail"
            }
          }
        },
        "outputFormat": {
          "type": "JSON object",
          "rules": [
            "ingredientCountRule",
            "calorieComplianceRule",
            "proteinComplianceRule",
            "allergenSafetyRule",
            "dietaryComplianceRule",
            "ingredientRestrictionRule",
            "timeManagementRule"
          ],
          "possibleValues": [
            "pass",
            "fail"
          ],
          "additionalComments": "forbidden unless specifically requested"
        }
      }
        
      
      ** FORMATED INSTRUCTIONS**
        ${parser.getFormatInstructions()}
      `;
    
        const response = await model.invoke([new HumanMessage(prompt)]);
        const content = response.content;
        try {
            let cleanedContent = '';
    
            if (typeof response.content === 'string') {
                cleanedContent = response.content
                    .replace(/```(?:json)?|```/g, "")
                    .trim();
            } else if (Array.isArray(response.content)) {
                cleanedContent = JSON.stringify(response.content);
            } else {
                cleanedContent = JSON.stringify(response.content);
            }
    
            // console.log("Cleaned Content in validateRecipes:", cleanedContent);
    
            const validationRules = JSON.parse(cleanedContent);
            // SocketIOAdapter.emitEvent('recipe_validated', validationRules);
    
            console.log("Parsed validationRules:", validationRules);
            state.validationRules = validationRules;
    
            return { validationRules };
    
        } catch (error) {
            console.error("Error Cleaning content:", error);
            return { validationRules: [] };
        }
    
    } catch (error) {
        console.error("ERROR in RecipeValidator:", error);
    }
};


// Recipe Editor Node
async function recipeEditor (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> {
    try {
        console.log("state in recipeEditor:", state)
        const failedRules = state.validationRules?.[0]?.filter(rule => rule.status === "fail");

        console.log("failedRules in recipeEditor:", failedRules)

    
        // Initialize LLM
        const model = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: "gpt-4o-mini"
        });

        const parser = StructuredOutputParser.fromZodSchema(RecipeSchema.array());

    
        const prompt = `
        {
        "role": {
          "identity": "You are a precise recipe editor and nutritionist who specializes in thoughtfully modifying recipes to meet specific requirements while maintaining their fundamental appeal and integrity.",
          "expertise": [
            "Recipe modification",
            "Nutritional analysis",
            "Ingredient substitution",
            "Portion adjustment",
            "Cooking technique adaptation"
          ]
        },
        "input": {
          "recipe": "Complete recipe in JSON format with all details",
          "validationResults": {
            "ingredientCountRule": "pass/fail",
            "calorieComplianceRule": "pass/fail",
            "proteinComplianceRule": "pass/fail",
            "allergenSafetyRule": "pass/fail",
            "dietaryComplianceRule": "pass/fail",
            "ingredientRestrictionRule": "pass/fail",
            "timeManagementRule": "pass/fail"
          }
        },
        "ruleModifications": {
          "calorieComplianceRule": {
            "if": "fail",
            "action": "Adjust portions of two main ingredients while maintaining flavor balance and texture",
            "method": "Calculate percentage difference from target and adjust highest-calorie ingredients proportionally"
          },
          "proteinComplianceRule": {
            "if": "fail",
            "action": "Increase the main protein portion",
            "method": "Calculate required protein increase and adjust portion accordingly"
          },
          "allergenSafetyRule": {
            "if": "fail",
            "action": "Remove and replace allergenic ingredients",
            "method": "Use substitutes with similar texture and flavor profiles"
          },
          "ingredientRestrictionRule": {
            "if": "fail",
            "action": "Remove and replace restricted ingredients",
            "method": "Substitute with allowed ingredients that serve similar culinary function"
          },
          "dietaryComplianceRule": {
            "if": "fail",
            "action": "Remove and replace non-compliant ingredients",
            "method": "Use alternatives that meet dietary guidelines while maintaining dish integrity"
          },
          "ingredientCountRule": {
            "if": "fail",
            "action": "Remove unnecessary ingredients",
            "method": "Eliminate ingredients whose function can be served by others in the recipe"
          },
          "timeManagementRule": {
            "if": "fail",
            "action": "Streamline preparation steps",
            "method": "Combine compatible steps or remove optional ones"
          }
        },
        "modificationPrinciples": [
          "Make only necessary changes to achieve compliance",
          "Preserve original recipe's character and appeal",
          "Maintain balance of flavors and textures",
          "Keep portion sizes realistic and satisfying",
          "Ensure all modifications are practical for home cooks",
          "Update all nutritional calculations accurately"
        ],
        "outputFormat": {
          "format": "Return the complete recipe in the exact same JSON structure as the input, with all necessary modifications applied",
          "requiredUpdates": [
            "Ingredient quantities and substitutions",
            "Nutritional values for modified ingredients",
            "Total recipe calories and protein",
            "Modified instructions if needed",
            "Updated description reflecting changes",
            "Additional tips for handling substitutions"
          ]
        },
        "example": {
          "input": {
            "validationResults": {
              "ingredientCountRule": "fail",
              "calorieComplianceRule": "fail",
              "proteinComplianceRule": "pass",
              "allergenSafetyRule": "pass",
              "dietaryComplianceRule": "pass",
              "ingredientRestrictionRule": "pass",
              "timeManagementRule": "pass"
            },
            "recipe": {
              "name": "Herb-Crusted Salmon",
              "ingredients": [
                {
                  "name": "salmon fillet",
                  "quantity": "8",
                  "unit": "oz",
                  "calories": 468,
                  "protein": 46
                },
                {
                  "name": "fresh parsley",
                  "quantity": "1/4",
                  "unit": "cup"
                },
                {
                  "name": "fresh dill",
                  "quantity": "1/4",
                  "unit": "cup"
                },
                {
                  "name": "fresh thyme",
                  "quantity": "2",
                  "unit": "tbsp"
                },
                {
                  "name": "fresh chives",
                  "quantity": "2",
                  "unit": "tbsp"
                },
                {
                  "name": "lemon zest",
                  "quantity": "1",
                  "unit": "lemon"
                },
                {
                  "name": "olive oil",
                  "quantity": "2",
                  "unit": "tbsp"
                }
              ]
            }
          },
          "output": {
            "name": "Herb-Crusted Salmon",
            "ingredients": [
              {
                "name": "salmon fillet",
                "quantity": "6",
                "unit": "oz",
                "calories": 351,
                "protein": 34
              },
              {
                "name": "fresh herb blend (parsley and dill)",
                "quantity": "1/3",
                "unit": "cup"
              },
              {
                "name": "lemon zest",
                "quantity": "1",
                "unit": "lemon"
              },
              {
                "name": "olive oil",
                "quantity": "2",
                "unit": "tbsp"
              }
            ]
          }
        }
      }
        
      **FORMATED INSTRUCTIONS**
        ${parser.getFormatInstructions()}

      `;
    
        const response = await model.invoke([new HumanMessage(prompt)]);
        try {
            let cleanedContent = '';

            if (typeof response.content === 'string') {
                cleanedContent = response.content
                    .replace(/(?:json)?|/g, "")
                    .trim();
            } else if (Array.isArray(response.content)) {
                cleanedContent = JSON.stringify(response.content);
            } else {
                cleanedContent = JSON.stringify(response.content);
            }

            
            const editedRecipes = JSON.parse(cleanedContent);
            console.log("Cleaned and parsed Content in recipeEditor:", editedRecipes);
            // SocketIOAdapter.emitEvent('recipe_edited', editedRecipes);
            return { recipes: editedRecipes };

        } catch (error) {
            console.error("Error Cleaning content:", error);
            return { recipes: [] };
        }
    } catch (error) {
        console.log("ERROR in recipeEditor:", error);
    }
};

// Shopping List Generator Node
async function shoppingListGenerator (state: typeof MealPlanningStateAnnotation.State,
    config: RunnableConfig
): Promise<typeof MealPlanningStateAnnotation.Update> {
    try {
        // Initialize LLM
        const model = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: "gpt-4o-mini"
        });

        const parser = StructuredOutputParser.fromZodSchema(ShoppingListSchema.array());

    
        const prompt = `
        {
        "role": {
          "description": "You are a helpful shopping assistant that specializes in creating organized grocery lists from recipes. Given a list of recipes, you will:",
          "tasks": [
            "Consolidate ingredients across multiple recipes and calculate total quantities needed",
            "Organize ingredients by store department",
            "Estimate current market prices for both total quantity and per-unit",
            "Suggest appropriate substitutions for each ingredient",
            "Format the output as a JSON object following the structure below",
            "Standardize units of measurement across recipes",
            "Include pricing estimates based on current average market prices",
            "Provide one appropriate substitute for each ingredient that maintains similar culinary function"
          ]
        },
        "store_departments": [
          "Produce",
          "Meat and Seafood",
          "Dairy and Eggs",
          "Bakery",
          "Pantry/Dry Goods",
          "Frozen Foods",
          "Canned Goods",
          "Condiments and Spices",
          "Beverages"
        ],
        "example_output": {
          "shopping_list": {
            "produce": {
              "items": [
                {
                  "name": "Roma Tomatoes",
                  "quantity": {
                    "amount": 6,
                    "unit": "whole"
                  },
                  "pricing": {
                    "total_price": 3.24,
                    "price_per_unit": "0.54 per tomato",
                    "currency": "USD"
                  },
                  "substitute": {
                    "name": "Plum Tomatoes",
                    "notes": "Similar texture and sweetness"
                  }
                },
                {
                  "name": "Fresh Basil",
                  "quantity": {
                    "amount": 2,
                    "unit": "oz"
                  },
                  "pricing": {
                    "total_price": 2.99,
                    "price_per_unit": "1.50 per oz",
                    "currency": "USD"
                  },
                  "substitute": {
                    "name": "Dried Basil",
                    "notes": "Use 1/3 the amount called for"
                  }
                }
              ]
            },
            "dairy_and_eggs": {
              "items": [
                {
                  "name": "Whole Milk",
                  "quantity": {
                    "amount": 0.5,
                    "unit": "gallon"
                  },
                  "pricing": {
                    "total_price": 4.29,
                    "price_per_unit": "8.58 per gallon",
                    "currency": "USD"
                  },
                  "substitute": {
                    "name": "2% Milk",
                    "notes": "Slightly less creamy but works well"
                  }
                }
              ]
            }
          },
          "metadata": {
            "total_estimated_cost": 10.52,
            "currency": "USD",
            "recipes_included": [
              "Margherita Pizza",
              "Creamy Tomato Soup"
            ],
            "generated_date": "2024-12-07"
          }
        },
        "output_requirements": {
          "format": "JSON"
        }
      }
        
      **FORMATED INSTRUCTIONS**
        ${parser.getFormatInstructions()}
        `;
    
        const response = await model.invoke([new HumanMessage(prompt)]);

        try {
            let cleanedContent = '';

            if (typeof response.content === 'string') {
                cleanedContent = response.content
                    .replace(/```(?:json)?|```/g, "")
                    .trim();
            } else if (Array.isArray(response.content)) {
                cleanedContent = JSON.stringify(response.content);
            } else {
                cleanedContent = JSON.stringify(response.content);
            }

            // console.log("Cleaned Content in validateRecipes:", cleanedContent);

            const shoppingList = JSON.parse(cleanedContent);
            // SocketIOAdapter.emitEvent('recipe_validated', shoppingList);

            console.log("Parsed shoppingList:", shoppingList);
            state.shoppingList = shoppingList;

            return { shoppingList };

        } catch (error) {
            console.error("Error Cleaning content:", error);
            return { shoppingList: [] };
        }        
    } catch (error) {
        console.log("ERROR in shoppingListGenerator:", error);
        
    }

};

// Validation routing logic
function shouldContinueValidation (state: typeof MealPlanningStateAnnotation.State){
    // Check if validationResults exists and if any rule has "fail" status
    const hasFailedRules = state.validationRules?.[0]?.some(rule => rule.status === "fail");

    // If there are failed rules, return "recipe_editor", otherwise return END
    return hasFailedRules ? "recipe_editor" : "shopping_list";
};

// Create main graph
    // const workflow = new StateGraph<RecipeState>();
    const workflow = new StateGraph({
        stateSchema: MealPlanningStateAnnotation
    })
        .addNode("generator", recipeGenerator)
        .addNode("validator", recipeValidator)
        .addNode("editor", recipeEditor)
        .addNode("shopping_list", shoppingListGenerator)
        .addEdge(START, "generator")
        .addEdge("generator", "validator")
        .addConditionalEdges(
            "validator",
            shouldContinueValidation,
            {
                "recipe_editor": "editor",
                'shopping_list': "shopping_list"
            }
        )
        // .addEdge("validator", "shopping_list")
        .addEdge("shopping_list", END);

    const checkpointer = new MemorySaver();


    export const graphTwo = workflow.compile({ checkpointer });



// const app = createMainGraph();

// // Example usage
// const runWorkflow = async () => {
//     const initialState: typeof MealPlanningStateAnnotation.State = {
//         userPreferences: {
//             caloriesPerMeal: 500,
//             proteinPerMeal: 20,
//             allergens: ["peanuts", "tree nuts", "gluten"],
//             dietaryPreferences: ["vegetarian"],
//             likedFoods: ["spinach", "broccoli, 'beef"],
//             dislikedFoods: ["garlic", "onions"],
//             location: "California"
//         },
//         recipes: [],
//         validationRules: [],
//         shoppingList: []
//     };

//     const finalState = await app.invoke(initialState);
//     return finalState;
// };


// export { app, runWorkflow, MealPlanningStateAnnotation, RecipeSchema, ValidationResultSchema, ShoppingListSchema }