"""
utils.py
Short explanations used in the app.
"""


def get_strategy_description(strategy: str) -> str:
    descriptions = {
        "Threshold Moving": (
            """
**Threshold Moving: Post-training adjustment**

Normally a model predicts the positive class when its confidence score is above 0.5.
This strategy changes that threshold based on your cost ratio.

If missing a target (FN) is much more expensive than a false alarm (FP),
the threshold is lowered: so the model flags more cases as positive,
catching more real cases even if it also produces more false alarms.

**Formula (Elkan, 2001):**
Optimal threshold = FP cost ÷ (FP cost + FN cost)

**Advantage:** No retraining needed. Fast and mathematically grounded.

**Limitation:** Relies on the model producing accurate probability scores.
            """
        ),
        "Class Weighting": (
            """
**Class Weighting, During-training adjustment**

This strategy tells the model during training that mistakes on one class
are more expensive than mistakes on the other.

The model adjusts its internal decision boundary so it penalises
the costly mistakes more heavily.

**How it works:** The cost ratio is passed directly to the classifier
as a class_weight parameter in scikit-learn.

**Advantage:** The model learns cost awareness from the start.

**Limitation:** Requires retraining when costs change.
            """
        ),
        "Resampling (SMOTE)": (
            """
**Resampling (SMOTE): Pre-training adjustment**

SMOTE stands for Synthetic Minority Oversampling Technique.
It creates new synthetic examples of the minority class before training,
so the model learns from a more balanced dataset.

**How it works:** SMOTE finds existing minority examples and generates
new ones by interpolating between them.

**Advantage:** Works with any classifier without modifying the model itself.

**Limitation:** Creates synthetic data that may not represent real cases.
Also slower because it changes the dataset before training.
            """
        ),
    }
    return descriptions.get(strategy, "")


def get_model_description(model_type: str) -> str:
    descriptions = {
        "Decision Tree": (
            "A tree-based model that makes decisions by asking a series of yes/no questions. "
            "Easy to understand and visualise. Good for seeing how decisions are made."
        ),
        "Logistic Regression": (
            "A linear model that estimates the probability of each class. "
            "Fast, simple, and gives well-calibrated probability scores. "
            "Best starting point for cost-sensitive learning."
        ),
        "Random Forest": (
            "An ensemble of many decision trees that vote on the final prediction. "
            "Usually more accurate and robust than a single tree. "
            "Good for complex datasets."
        ),
        "SVM": (
            "Support Vector Machine. Finds the best boundary between classes. "
            "Can model complex, non-linear patterns. "
            "Slower to train but powerful on difficult problems."
        ),
    }
    return descriptions.get(model_type, "")


def get_strategy_story(strategy: str) -> str:
    """A story-mode explanation of each strategy, using a 'security guard' analogy."""
    stories = {
        "Threshold Moving": (
            """
**🕵️ The Story: Adjusting the Guard's Trigger Finger**

Picture a security guard who's already finished training, fully qualified, doing the job exactly the
way they were taught. By default, they only stop someone if they're more than 50% sure something's
wrong.

One day you pull the guard aside and say: *"Listen, missing a real problem here is way more expensive
than one extra false alarm. From now on, if you're even 30% sure, stop them anyway."*

Nothing about the guard's brain changed. They didn't relearn the job. You just moved their trigger
finger, using the exact cost ratio you set on the left, to match how expensive each type of mistake
really is. It's the fastest fix of the three: no retraining, just a new instruction.
            """
        ),
        "Class Weighting": (
            """
**🕵️ The Story: Briefing the Guard on Day One**

This time, you catch the guard before training even begins. You sit them down and say: *"Here's the
deal: missing a real problem is ten times worse than a false alarm. Every time you get quizzed during
training, getting the rare, costly case wrong will sting ten times harder than a false alarm."*

So throughout the entire training process, every mistake on the costly case is treated as a much
bigger deal. By the time training's done, the guard's whole instinct for the job has been shaped
around that priority, not bolted on afterward like with Threshold Moving.

The catch: if your cost ratio changes later, you can't just tweak a dial. You have to retrain the
guard from scratch with the new briefing.
            """
        ),
        "Resampling (SMOTE)": (
            """
**🕵️ The Story: Extra Practice Drills**

Before training starts, you look at the guard's practice material and realize something: they've
barely ever seen the rare, costly case. Out of a thousand practice scenarios, maybe five were the
real threat. No wonder it feels like a bizarre, once-in-a-blue-moon fluke to them.

So you create extra practice drills. Not by copy-pasting the same five examples over and over
(that would just teach them to memorise those exact five), but by generating realistic *variations*
blended from the real rare cases, synthetic near-misses that still feel authentic.

By the time training starts for real, the rare case doesn't feel rare to the guard anymore, it feels
familiar. The trade-off: building all those extra drills takes time, and they're still practice
scenarios, not the real thing.
            """
        ),
    }
    return stories.get(strategy, "")


def get_model_story(model_type: str) -> str:
    """A story-mode explanation of each model, framed as a type of security guard."""
    stories = {
        "Logistic Regression": (
            """
**🧑‍⚖️ Meet: The Calm Judge**

This guard looks at all the evidence and draws exactly one straight line in the sand: everything on
this side is "fine," everything on that side is "risky." No exceptions, no special cases, just one
clean rule.

The Calm Judge never panics and never overthinks. They're fast, and unlike some guards, they'll
happily tell you *how* confident they are, not just yes or no. That's exactly what lets Threshold
Moving work so well with this model.

**Their blind spot:** if the real boundary between "risky" and "fine" is curvy or complicated, one
straight line just can't capture it.
            """
        ),
        "Decision Tree": (
            """
**🌳 Meet: The Twenty Questions Guard**

This guard decides things the way you'd play Twenty Questions. *"Is the amount over $500? Yes.
Is it a new device? Yes. Flag it."* Follow the branches down and you land on a decision.

Their biggest strength: you can ask them *why* they made a call, and they'll show you the exact
chain of questions. Nothing hidden, nothing mysterious.

**Their blind spot:** left alone, a single Twenty Questions Guard can get a bit too clever, memorising
quirks specific to their exact training data instead of learning the general pattern. That's part of
why the next guard exists.
            """
        ),
        "Random Forest": (
            """
**🌲🌲🌲 Meet: The Committee**

Instead of trusting one Twenty Questions Guard, you hire a hundred of them, each one trained on a
slightly different slice of the practice data, and let them vote on every decision.

Any single guard on the committee might have a slightly odd opinion. But when a hundred of them vote
together, those individual quirks tend to cancel out, and the majority verdict is usually more
reliable than any one guard alone.

**The trade-off:** you can't ask "why" as cleanly anymore, since the answer is now "because 67 out of
100 guards agreed," not one clean chain of questions. It's also slower, since you're really running a
hundred guards at once.
            """
        ),
        "SVM": (
            """
**✂️ Meet: The Perfectionist**

This guard refuses to draw a lazy line. Instead, they search for the *single best possible* boundary,
the one that leaves as much breathing room as possible between "risky" and "fine," and if a straight
line won't cut it, they'll bend that boundary into a curve to fit the data properly.

Because they're so thorough, the Perfectionist can catch subtle patterns that a simpler guard like
the Calm Judge would walk straight past.

**The trade-off:** all that care takes time, especially with a lot of data, and they're the slowest
of the four guards to train.
            """
        ),
    }
    return stories.get(model_type, "")

