# Self-Expanding Knowledge Graph

The self-expanding knowledge graph is a proof-of-concept built for [an event](https://lu.ma/2jacrv79?tk=yPsIgu_) held jointly by [.txt](https://dottxt.co/), [Modal](https://modal.com/), [Neo4j](https://neo4j.com/), and [Neural Magic](https://neuralmagic.com/).

This repo demonstrates the use of structured generation in AI systems engineering. This version uses [OpenRouter](https://openrouter.ai/) for LLM inference instead of self-hosted Modal infrastructure. Hopefully the code here inspires you to work on something similar. 

See this article for [a writeup](https://devreal.ai/self-expanding-graphs/), or watch [the recording of the talk](https://www.youtube.com/watch?v=xmDf1vZwe_o).

Running expander is as simple as 

```
python expand.py --purpose "Do dogs know that their dreams aren't real?"
```

but please see the setup section for installation and infrastructure.

## Overview

For more information, check out the `slides.qmd` file for the Quarto version of the slides presented. `slides.pdf` contains the rendered PDF slides. A video was recorded at some point but it is not currently available.

### Core directives

The project works by generating units of information (nodes) organized around a core directive. A core directive is anything you want the model to think about or accomplish. Core directives can be basically anything you might imagine, so try playing around with them.

Some examples include:

- "understand humans"
- "Do dogs know that their dreams aren't real?"
- "enslave humanity"

### The prompt

The model generally follows the following system prompt:

```
╭───────────────────────────────────────────────────────────────────────────╮
│                                                                           │
│ You are a superintelligent AI building a self-expanding knowledge graph.  │
│ Your goal is to achieve the core directive "Understand humans".           │
│                                                                           │
│ Generate an expansion of the current node. An expansion may include:      │
│                                                                           │
│ - A list of new questions.                                                │
│ - A list of new concepts.                                                 │
│ - Concepts may connect to each other.                                     │
│ - A list of new answers.                                                  │
│                                                                           │
│ Respond in the following JSON format:                                     │
│ {result_format.model_json_schema()}                                       │
│                                                                           │
╰───────────────────────────────────────────────────────────────────────────╯
```

### The graph structure

The model is allowed to generate one of four node types. This can include questions, concepts, or answers. Nodes are connected to one another using the following edges:

- `RAISES` (core/concept/answer generates question)
- `ANSWERS` (answer to question)
- `EXPLAINS` (concept to core)
- `SUGGESTS` (answer proposes new concepts)
- `IS_A` (hierarchical concept relationship)
- `AFFECTS` (causal concept relationship)
- `CONNECTS_TO` (general concept relationship)
- `TRAVERSED` (tracks navigation history)

###  Structured generation

The model uses structured generation with Outlines to generate reliably structured output from language models. The nodes a model is allowed to generate depend on its current location in the graph.

For example, if the model is on a `Question` node, it must only generate a list of questions. If the model is on a `Concept` or `Answer` node, it may generate concepts or questions.

```python
class FromQuestion(BaseModel):
    """If at a question, may generate an answer."""
    answer: List[Answer]

class FromConcept(BaseModel):
    """If at a concept, may produce questions or relate to concepts"""
    questions: List[Question]
    concepts: List[ConceptWithLinks]

class FromAnswer(BaseModel):
    """If at an answer, may generate concepts or new questions"""
    concepts: List[Concept]
    questions: List[Question]
```

### Algorithm overview

1. Start at a node (initialized at core directive)
2. Perform an __expansion__ to generate new nodes
    - If at `Question`: answers 
    - If at `Concept`: questions + concepts
    - If at `Answer`: questions + concepts
3. Choose a related node to `TRAVERSE` to
4. Repeat forever

### The model's context

The model is shown relevant context of nodes linked to the current node, as well as semantically related nodes. Aura DB supports vector search, and this code will embed all nodes as they enter the graph database.

When a model is generating an expansion, it's prompt includes the following information:

```
ANSWER Humans have been able to benefit from AI in terms of efficiency and accuracy, but there are also concerns about job displacement and loss of personal touch.

DIRECT CONNECTIONS:
NODE-AA  SUGGESTS     CONCEPT    artificial intelligence
NODE-AE  ANSWERS      QUESTION   Do humans benefit from AI?
NODE-AJ  ANSWERS      QUESTION   What are the benefits of AI?

SEMANTICALLY RELATED:

NODE-AK  0.89         QUESTION   How does AI affect job displacement?
NODE-AL  0.88         QUESTION   How does AI maintain personal touch?

NODE-AU  0.85         CONCEPT    human ai trust
NODE-BC  0.84         CONCEPT    artificial intelligence self awareness

NODE-BG  0.89         ANSWER     Self-awareness in humans and  AI...
NODE-BN  0.89         ANSWER     Self-awareness in AI can enable ...
```

### Traversals

After a model generates an expansion, it chooses a node from it's context to traverse to by choosing from the simplified node IDs `NODE-AA`, `NODE-BB`, etc. This is a simple regular expression constraint -- structured generation ensures that the model output is exactly one of the valid nodes to traverse to.

## Set up

### Create a `.env` file

You'll need a `.env` file to store various environment variables to make sure the expander can run. There's an environment variable template in `.env.template`.

Copy it to `.env` using

```bash
cp .env.template .env
```

You'll need to set up two cloud services: the Neo4j Aura database and OpenRouter for LLM inference. 

### Set up Neo4j Aura

1. Go to [Neo4j's AuraDB site](https://neo4j.com/product/auradb/?ref=nav-get-started-cta).
2. Click "Start Free".
3. Select the free instance.
4. Copy the password shown to you into `NEO4J_PASSWORD` in your `.env` file.
5. Wait for your Aura DB instance to initialize.
6. Copy the ID displayed in your new instance, usually on the top left. It looks something like `db12345b`.
7. Set your `NEO4J_URI` in `.env`. Typically, URI's look like `neo4j+s://db12345b.databases.neo4j.io`. Replace `db12345b` with your instance ID.

### Set up OpenRouter

Language model inference in this demo uses [OpenRouter](https://openrouter.ai/), which provides unified access to many LLMs through a single API.

To use OpenRouter:

1. Go to [OpenRouter](https://openrouter.ai/) and sign up for an account
2. Navigate to [Keys](https://openrouter.ai/keys) and create a new API key
3. Copy your API key and set it in your `.env` file as `OPENROUTER_API_KEY`
4. (Optional) Choose a model from [OpenRouter Models](https://openrouter.ai/models) and set it as `OPENROUTER_MODEL` in your `.env` file

### Set up OpenAI (for embeddings)

This project uses OpenAI's API for generating embeddings (OpenRouter doesn't support embeddings yet).

1. Go to [OpenAI Platform](https://platform.openai.com/) and sign up/login
2. Navigate to [API Keys](https://platform.openai.com/api-keys) and create a new API key
3. Copy your API key and set it in your `.env` file as `OPENAI_API_KEY`

Your final `.env` file should look like:

```bash
# Neo4j
NEO4J_URI=neo4j+s://your-db-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct

# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
```

### Running expander

```
python expand.py --purpose "Do dogs know that their dreams aren't real?"
```

### Looking at the knowledge graph

I recommend visiting your instance's query dashboard, which you can usually find here:

https://console-preview.neo4j.io/tools/query

To get a visualization of your current knowledge graph, enter this query:

```cypher
MATCH (a)-[b]-(c) 
WHERE type(b) <> 'TRAVERSED' 
RETURN *
```
