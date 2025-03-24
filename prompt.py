# prompt_template = """Task: Generate keywords from the following incomplete code snippet. The keywords should focus on the functionality, tech stack, or code logic, and will be used for code completion tasks.
# ### code snippet
# {code}

# ### output
# Output Format: Only output separate keywords with commas.
# Output template: class, method, loop, string, object, join, regex, assert, module, function
# Output:
# """

# prompt_template = """Task: Extract technical keywords from the code snippet below.

# Requirements:
# 1. ONLY output keywords separated by commas
# 2. Focus on programming concepts, functions, and patterns present in the code
# 3. DO NOT explain the code or provide any additional text
# 4. Keywords should be single words or short technical terms
# 5. Expected Output Format: keyword1, keyword2, keyword3, ...

# Code Snippet:
# {code}

# Keywords:
# """

# rag_retrieval_prompt_template = """Task: Extract key technical terms and concepts from the incomplete code snippet below for retrieval purposes.

# Instructions:
# 1. Analyze the given code snippet carefully.
# 2. Identify important programming concepts, functions, libraries, and patterns present in the code.
# 3. Extract keywords that best represent the technical aspects and context of the code.
# 4. Output ONLY the keywords, separated by commas.
# 5. Focus on terms that would be most useful for retrieving relevant documentation or similar code examples.
# 6. DO NOT explain the code or provide any additional text.
# 7. Keywords should be single words or short technical phrases.

# Code Snippet:
# {code}

# Keywords:
# """

# enhanced_prompt = """Task: Extract precise technical keywords from code snippet for code completion retrieval.

# Technical Requirements:
# 1. Extract ONLY programming-specific terms (libraries, functions, patterns, etc)
# 2. Prioritize in this order:
#    a) Explicitly mentioned APIs/modules
#    b) Core programming constructs
#    c) Implied technical context
# 3. Filter criteria:
#    - Exclude generic terms like 'class', 'method' unless contextually specific
#    - Include parameter types if significant
#    - Capture framework-specific patterns
# 4. Format: Strictly comma-separated lowercase terms, no explanations

# Code Snippet:
# {code}

# Extracted Keywords:
# """

prompt_template = """Task: Extract precise technical keywords from this incomplete code snippet for code completion retrieval.

Requirements:
1. ONLY output keywords separated by commas (no other text)
2. Include:
   - Language-specific features
   - Libraries/frameworks/modules used
   - Data structures
   - Algorithms or patterns
   - Function signatures and parameter types
   - Class relationships and inheritance patterns
3. Prioritize specific technical terms over general programming concepts
4. Include domain-specific terminology if present
5. 5-15 keywords is ideal, focus on quality over quantity

Code Snippet:
{code}

Keywords:
"""