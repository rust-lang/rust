Graph compression

The graph compression algorithm is intended to remove and minimize the
size of the dependency graph so it can be saved, while preserving
everything we care about. In particular, given a set of input/output
nodes in the graph (which must be disjoint), we ensure that the set of
input nodes that can reach a given output node does not change,
although the intermediate nodes may change in various ways. In short,
the output nodes are intended to be the ones whose existence we care
about when we start up, because they have some associated data that we
will try to re-use (and hence if they are dirty, we have to throw that
data away). The other intermediate nodes don't really matter so much.

### Overview

The algorithm works as follows:

1. Do a single walk of the graph to construct a DAG
    - in this walk, we identify and unify all cycles, electing a representative "head" node
    - this is done using the union-find implementation
    - this code is found in the `classify` module
2. The result from this walk is a `Dag`: 
   - the set of SCCs, represented by the union-find table
   - a set of edges in the new DAG, represented by:
     - a vector of parent nodes for each child node
     - a vector of cross-edges
     - once these are canonicalized, some of these edges may turn out to be cyclic edges
       (i.e., an edge A -> A where A is the head of some SCC)
3. We pass this `Dag` into the construct code, which then creates a
   new graph.  This graph has a smaller set of indices which includes
   *at least* the inputs/outputs from the original graph, but may have
   other nodes as well, if keeping them reduces the overall size of
   the graph.
   - This code is found in the `construct` module.
   
### Some notes

The input graph is assumed to have *read-by* edges. i.e., `A -> B`
means that the task B reads data from A. But the DAG defined by
classify is expressed in terms of *reads-from* edges, which are the
inverse. So `A -> B` is the same as `B -rf-> A`. *reads-from* edges
are more natural since we want to walk from the outputs to the inputs,
effectively. When we construct the final graph, we reverse these edges
back into the *read-by* edges common elsewhere.

   
   
   
