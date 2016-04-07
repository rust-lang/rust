This is the code to load/save the dependency graph. Loading is assumed
to run early in compilation, and saving at the very end. When loading,
the basic idea is that we will load up the dependency graph from the
previous compilation and compare the hashes of our HIR nodes to the
hashes of the HIR nodes that existed at the time. For each node whose
hash has changed, or which no longer exists in the new HIR, we can
remove that node from the old graph along with any nodes that depend
on it. Then we add what's left to the new graph (if any such nodes or
edges already exist, then there would be no effect, but since we do
this first thing, they do not).



