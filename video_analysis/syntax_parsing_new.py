# import requests, json
# text = "Come join us for our back-to-campus event to connect with other HSM members and learn more about what we do!"
# r = requests.post(
#     "http://localhost:9000/",
#     params={'annotators': 'parse', 'outputFormat': 'json'},
#     data=text.encode('utf-8')
# )
# tree = r.json()['sentences'][0]['parse']
# print(tree)


# Basic stanza example
# import numpy as np
# import stanza
# stanza.download('en') # download English model
# nlp = stanza.Pipeline('en') # initialize English neural pipeline
# doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence

# for sentence in doc.sentences:
#     for word in sentence.words:
#         print(word.text, word.lemma, word.pos)

# for sentence in doc.sentences:
#     print(sentence.ents)
#     print(sentence.dependencies)



# Stanza example for calculating syntactic complexity
import stanza
from nltk import Tree

from graphviz import Digraph

# Function to visualize the tree using graphviz
def visualize_tree(tree):
    dot = Digraph()
    def add_nodes_edges(t, parent=None):
        node_id = str(id(t))
        label = t.label() if isinstance(t, Tree) else t
        dot.node(node_id, label)
        if parent:
            dot.edge(parent, node_id)
        if isinstance(t, Tree):
            for child in t:
                add_nodes_edges(child, node_id)
    add_nodes_edges(tree)
    return dot

# Function to get constituent boundaries for pauses
def get_pause_positions(tree, labels=('S', 'SBAR', 'VP', 'NP')):
    """
    Return a list of (index, word) tuples marking where to insert pauses.
    """
    words = tree.leaves()
    pause_indices = []

    def traverse(node, start_idx=0):
        # Leaf node → single token span
        if isinstance(node[0], str):
            return 1

        total_span = 0
        for child in node:
            span = traverse(child, start_idx + total_span)
            total_span += span

        if node.label() in labels:
            pause_indices.append(start_idx + total_span - 1)

        return total_span

    traverse(tree)

    # Return both index and corresponding word
    pauses = [(i, words[i]) for i in sorted(set(pause_indices)) if i < len(words)]
    return pauses

# stanza.download("en")
nlp = stanza.Pipeline(lang="en")

# Functions to calculate tree depth and clause count
def get_tree_depth(tree):
    if tree.children is None or len(tree.children) == 0:
        return 1
    return 1 + max(get_tree_depth(child) for child in tree.children)

def count_clauses(tree):
    return sum(1 for node in tree.nodes if node.label in ("S", "SBAR"))

# Sample text for analysis
text = "Yeah, so I'm not sure what exactly this is called, but it's a little gift that a friend of mine gave me at the end of Great 9 when she was going back to Japan for school Yeah, I think it's some kind of origami looks super awesome and Yeah, I've kept it ever since on my desk because Yeah, I really like to hold on to souvenirs from the past and Yeah, it just reminds me of all the good times we had all the memories that we shared Yeah, I don't know I'm a big hoarder of souvenirs. I have like a whole box lying around with every possible item you can imagine Anyways, um Yeah, let me know in the comments if You know what this is called because I would love to find out, but I don't know where my friend is now Yeah"

# Process the text
doc = nlp(text)
depths = []
clauses = []

for sentence in doc.sentences:
    tree = sentence.constituency
    print(tree)
    nltk_tree = Tree.fromstring(str(tree))
    nltk_tree.pretty_print()
    dot = visualize_tree(nltk_tree)
    dot.render(f'syntax_tree{sentence.index}', format='png', cleanup=True)
    pauses = get_pause_positions(nltk_tree)
    print("Sentence:", " ".join(nltk_tree.leaves()))
    print("Pauses (index, word):", pauses)
    print()
    # depths.append(get_tree_depth(tree))
    # clauses.append(count_clauses(tree))




# print("Average Tree Depth:", sum(depths)/len(depths))
# print("Average Clause Count:", sum(clauses)/len(clauses))