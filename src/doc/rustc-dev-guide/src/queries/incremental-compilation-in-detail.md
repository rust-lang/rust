# Incremental compilation in detail

The incremental compilation scheme is, in essence, a surprisingly
simple extension to the overall query system. It relies on the fact that:

  1. queries are pure functions -- given the same inputs, a query will always
     yield the same result, and
  2. the query model structures compilation in an acyclic graph that makes
     dependencies between individual computations explicit.

This chapter will explain how we can use these properties for making things
incremental and then goes on to discuss version implementation issues.

## A Basic Algorithm For Incremental Query Evaluation

As explained in the [query evaluation model primer][query-model], query
invocations form a directed-acyclic graph. Here's the example from the
previous chapter again:

```ignore
  list_of_all_hir_items <----------------------------- type_check_crate()
                                                               |
                                                               |
  Hir(foo) <--- type_of(foo) <--- type_check_item(foo) <-------+
                                      |                        |
                    +-----------------+                        |
                    |                                          |
                    v                                          |
  Hir(bar) <--- type_of(bar) <--- type_check_item(bar) <-------+
```

Since every access from one query to another has to go through the query
context, we can record these accesses and thus actually build this dependency
graph in memory. With dependency tracking enabled, when compilation is done,
we know which queries were invoked (the nodes of the graph) and for each
invocation, which other queries or input has gone into computing the query's
result (the edges of the graph).

Now suppose we change the source code of our program so that
HIR of `bar` looks different than before. Our goal is to only recompute
those queries that are actually affected by the change while re-using
the cached results of all the other queries. Given the dependency graph we can
do exactly that. For a given query invocation, the graph tells us exactly
what data has gone into computing its results, we just have to follow the
edges until we reach something that has changed. If we don't encounter
anything that has changed, we know that the query still would evaluate to
the same result we already have in our cache.

Taking the `type_of(foo)` invocation from above as an example, we can check
whether the cached result is still valid by following the edges to its
inputs. The only edge leads to `Hir(foo)`, an input that has not been affected
by the change. So we know that the cached result for `type_of(foo)` is still
valid.

The story is a bit different for `type_check_item(foo)`: We again walk the
edges and already know that `type_of(foo)` is fine. Then we get to
`type_of(bar)` which we have not checked yet, so we walk the edges of
`type_of(bar)` and encounter `Hir(bar)` which *has* changed. Consequently
the result of `type_of(bar)` might yield a different result than what we
have in the cache and, transitively, the result of `type_check_item(foo)`
might have changed too. We thus re-run `type_check_item(foo)`, which in
turn will re-run `type_of(bar)`, which will yield an up-to-date result
because it reads the up-to-date version of `Hir(bar)`. Also, we re-run
`type_check_item(bar)` because result of `type_of(bar)` might have changed.


## The problem with the basic algorithm: false positives

If you read the previous paragraph carefully you'll notice that it says that
`type_of(bar)` *might* have changed because one of its inputs has changed.
There's also the possibility that it might still yield exactly the same
result *even though* its input has changed. Consider an example with a
simple query that just computes the sign of an integer:

```ignore
  IntValue(x) <---- sign_of(x) <--- some_other_query(x)
```

Let's say that `IntValue(x)` starts out as `1000` and then is set to `2000`.
Even though `IntValue(x)` is different in the two cases, `sign_of(x)` yields
the result `+` in both cases.

If we follow the basic algorithm, however, `some_other_query(x)` would have to
(unnecessarily) be re-evaluated because it transitively depends on a changed
input. Change detection yields a "false positive" in this case because it has
to conservatively assume that `some_other_query(x)` might be affected by that
changed input.

Unfortunately it turns out that the actual queries in the compiler are full
of examples like this and small changes to the input often potentially affect
very large parts of the output binaries. As a consequence, we had to make the
change detection system smarter and more accurate.

## Improving accuracy: the red-green algorithm

The "false positives" problem can be solved by interleaving change detection
and query re-evaluation. Instead of walking the graph all the way to the
inputs when trying to find out if some cached result is still valid, we can
check if a result has *actually* changed after we were forced to re-evaluate
it.

We call this algorithm the red-green algorithm because nodes
in the dependency graph are assigned the color green if we were able to prove
that its cached result is still valid and the color red if the result has
turned out to be different after re-evaluating it.

The meat of red-green change tracking is implemented in the try-mark-green
algorithm, that, you've guessed it, tries to mark a given node as green:

```rust,ignore
fn try_mark_green(tcx, current_node) -> bool {

    // Fetch the inputs to `current_node`, i.e. get the nodes that the direct
    // edges from `node` lead to.
    let dependencies = tcx.dep_graph.get_dependencies_of(current_node);

    // Now check all the inputs for changes
    for dependency in dependencies {

        match tcx.dep_graph.get_node_color(dependency) {
            Green => {
                // This input has already been checked before and it has not
                // changed; so we can go on to check the next one
            }
            Red => {
                // We found an input that has changed. We cannot mark
                // `current_node` as green without re-running the
                // corresponding query.
                return false
            }
            Unknown => {
                // This is the first time we look at this node. Let's try
                // to mark it green by calling try_mark_green() recursively.
                if try_mark_green(tcx, dependency) {
                    // We successfully marked the input as green, on to the
                    // next.
                } else {
                    // We could *not* mark the input as green. This means we
                    // don't know if its value has changed. In order to find
                    // out, we re-run the corresponding query now!
                    tcx.run_query_for(dependency);

                    // Fetch and check the node color again. Running the query
                    // has forced it to either red (if it yielded a different
                    // result than we have in the cache) or green (if it
                    // yielded the same result).
                    match tcx.dep_graph.get_node_color(dependency) {
                        Red => {
                            // The input turned out to be red, so we cannot
                            // mark `current_node` as green.
                            return false
                        }
                        Green => {
                            // Re-running the query paid off! The result is the
                            // same as before, so this particular input does
                            // not invalidate `current_node`.
                        }
                        Unknown => {
                            // There is no way a node has no color after
                            // re-running the query.
                            panic!("unreachable")
                        }
                    }
                }
            }
        }
    }

    // If we have gotten through the entire loop, it means that all inputs
    // have turned out to be green. If all inputs are unchanged, it means
    // that the query result corresponding to `current_node` cannot have
    // changed either.
    tcx.dep_graph.mark_green(current_node);

    true
}
```

> NOTE:
> The actual implementation can be found in
> [`compiler/rustc_query_system/src/dep_graph/graph.rs`][try_mark_green]

By using red-green marking we can avoid the devastating cumulative effect of
having false positives during change detection. Whenever a query is executed
in incremental mode, we first check if its already green. If not, we run
`try_mark_green()` on it. If it still isn't green after that, then we actually
invoke the query provider to re-compute the result. Re-computing the query might 
then itself involve recursively invoking more queries, which can mean we come back
to the `try_mark_green()` algorithm for the dependencies recursively.


## The real world: how persistence makes everything complicated

The sections above described the underlying algorithm for incremental
compilation but because the compiler process exits after being finished and
takes the query context with its result cache with it into oblivion, we have to
persist data to disk, so the next compilation session can make use of it.
This comes with a whole new set of implementation challenges:

- The query result cache is stored to disk, so they are not readily available
  for change comparison.
- A subsequent compilation session will start off with new version of the code
  that has arbitrary changes applied to it. All kinds of IDs and indices that
  are generated from a global, sequential counter (e.g. `NodeId`, `DefId`, etc)
  might have shifted, making the persisted results on disk not immediately
  usable anymore because the same numeric IDs and indices might refer to
  completely new things in the new compilation session.
- Persisting things to disk comes at a cost, so not every tiny piece of
  information should be actually cached in between compilation sessions.
  Fixed-sized, plain-old-data is preferred to complex things that need to run
  through an expensive (de-)serialization step.

The following sections describe how the compiler solves these issues.

### A Question Of Stability: Bridging The Gap Between Compilation Sessions

As noted before, various IDs (like `DefId`) are generated by the compiler in a
way that depends on the contents of the source code being compiled. ID assignment
is usually deterministic, that is, if the exact same code is compiled twice,
the same things will end up with the same IDs. However, if something
changes, e.g. a function is added in the middle of a file, there is no
guarantee that anything will have the same ID as it had before.

As a consequence we cannot represent the data in our on-disk cache the same
way it is represented in memory. For example, if we just stored a piece
of type information like `TyKind::FnDef(DefId, &'tcx Substs<'tcx>)` (as we do
in memory) and then the contained `DefId` points to a different function in
a new compilation session we'd be in trouble.

The solution to this problem is to find "stable" forms for IDs which remain
valid in between compilation sessions. For the most important case, `DefId`s,
these are the so-called `DefPath`s. Each `DefId` has a
corresponding `DefPath` but in place of a numeric ID, a `DefPath` is based on
the path to the identified item, e.g. `std::collections::HashMap`. The
advantage of an ID like this is that it is not affected by unrelated changes.
For example, one can add a new function to `std::collections` but
`std::collections::HashMap` would still be `std::collections::HashMap`. A
`DefPath` is "stable" across changes made to the source code while a `DefId`
isn't.

There is also the `DefPathHash` which is just a 128-bit hash value of the
`DefPath`. The two contain the same information and we mostly use the
`DefPathHash` because it simpler to handle, being `Copy` and self-contained.

This principle of stable identifiers is used to make the data in the on-disk
cache resilient to source code changes. Instead of storing a `DefId`, we store
the `DefPathHash` and when we deserialize something from the cache, we map the
`DefPathHash` to the corresponding `DefId` in the *current* compilation session
(which is just a simple hash table lookup).

The `HirId`, used for identifying HIR components that don't have their own
`DefId`, is another such stable ID. It is (conceptually) a pair of a `DefPath`
and a `LocalId`, where the `LocalId` identifies something (e.g. a `hir::Expr`)
locally within its "owner" (e.g. a `hir::Item`). If the owner is moved around,
the `LocalId`s within it are still the same.



### Checking query results for changes: `HashStable` and `Fingerprint`s

In order to do red-green-marking we often need to check if the result of a
query has changed compared to the result it had during the previous
compilation session. There are two performance problems with this though:

- We'd like to avoid having to load the previous result from disk just for
  doing the comparison. We already computed the new result and will use that.
  Also loading a result from disk will "pollute" the interners with data that
  is unlikely to ever be used.
- We don't want to store each and every result in the on-disk cache. For
  example, it would be wasted effort to persist things to disk that are
  already available in upstream crates.

The compiler avoids these problems by using so-called `Fingerprint`s. Each time
a new query result is computed, the query engine will compute a 128 bit hash
value of the result. We call this hash value "the `Fingerprint` of the query
result". The hashing is (and has to be) done "in a stable way". This means
that whenever something is hashed that might change in between compilation
sessions (e.g. a `DefId`), we instead hash its stable equivalent
(e.g. the corresponding `DefPath`). That's what the whole `HashStable`
infrastructure is for. This way `Fingerprint`s computed in two
different compilation sessions are still comparable.

The next step is to store these fingerprints along with the dependency graph.
This is cheap since fingerprints are just bytes to be copied. It's also cheap to
load the entire set of fingerprints together with the dependency graph.

Now, when red-green-marking reaches the point where it needs to check if a
result has changed, it can just compare the (already loaded) previous
fingerprint to the fingerprint of the new result.

This approach works rather well but it's not without flaws:

- There is a small possibility of hash collisions. That is, two different
  results could have the same fingerprint and the system would erroneously
  assume that the result hasn't changed, leading to a missed update.

  We mitigate this risk by using a high-quality hash function and a 128 bit
  wide hash value. Due to these measures the practical risk of a hash
  collision is negligible.

- Computing fingerprints is quite costly. It is the main reason why incremental
  compilation can be slower than non-incremental compilation. We are forced to
  use a good and thus expensive hash function, and we have to map things to
  their stable equivalents while doing the hashing.


### A tale of two `DepGraph`s: the old and the new

The initial description of dependency tracking glosses over a few details
that quickly become a head scratcher when actually trying to implement things.
In particular it's easy to overlook that we are actually dealing with *two*
dependency graphs: The one we built during the previous compilation session and
the one that we are building for the current compilation session.

When a compilation session starts, the compiler loads the previous dependency
graph into memory as an immutable piece of data. Then, when a query is invoked,
it will first try to mark the corresponding node in the graph as green. This
means really that we are trying to mark the node in the *previous* dep-graph
as green that corresponds to the query key in the *current* session. How do we
do this mapping between current query key and previous `DepNode`? The answer
is again `Fingerprint`s: Nodes in the dependency graph are identified by a
fingerprint of the query key. Since fingerprints are stable across compilation
sessions, computing one in the current session allows us to find a node
in the dependency graph from the previous session. If we don't find a node with
the given fingerprint, it means that the query key refers to something that
did not yet exist in the previous session.

So, having found the dep-node in the previous dependency graph, we can look
up its dependencies (i.e. also dep-nodes in the previous graph) and continue with
the rest of the try-mark-green algorithm. The next interesting thing happens
when we successfully marked the node as green. At that point we copy the node
and the edges to its dependencies from the old graph into the new graph. We
have to do this because the new dep-graph cannot acquire the
node and edges via the regular dependency tracking. The tracking system can
only record edges while actually running a query -- but running the query,
although we have the result already cached, is exactly what we want to avoid.

Once the compilation session has finished, all the unchanged parts have been
copied over from the old into the new dependency graph, while the changed parts
have been added to the new graph by the tracking system. At this point, the
new graph is serialized out to disk, alongside the query result cache, and can
act as the previous dep-graph in a subsequent compilation session.


### Didn't you forget something?: cache promotion

The system described so far has a somewhat subtle property: If all inputs of a
dep-node are green then the dep-node itself can be marked as green without
computing or loading the corresponding query result. Applying this property
transitively often leads to the situation that some intermediate results are
never actually loaded from disk, as in the following example:

```ignore
   input(A) <-- intermediate_query(B) <-- leaf_query(C)
```

The compiler might need the value of `leaf_query(C)` in order to generate some
output artifact. If it can mark `leaf_query(C)` as green, it will load the
result from the on-disk cache. The result of `intermediate_query(B)` is never
loaded though. As a consequence, when the compiler persists the *new* result
cache by writing all in-memory query results to disk, `intermediate_query(B)`
will not be in memory and thus will be missing from the new result cache.

If there subsequently is another compilation session that actually needs the
result of `intermediate_query(B)` it will have to be re-computed even though we
had a perfectly valid result for it in the cache just before.

In order to prevent this from happening, the compiler does something called
"cache promotion": Before emitting the new result cache it will walk all green
dep-nodes and make sure that their query result is loaded into memory. That way
the result cache doesn't unnecessarily shrink again.



# Incremental compilation and the compiler backend

The compiler backend, the part involving LLVM, is using the query system but
it is not implemented in terms of queries itself. As a consequence it does not
automatically partake in dependency tracking. However, the manual integration
with the tracking system is pretty straight-forward. The compiler simply tracks
what queries get invoked when generating the initial LLVM version of each
codegen unit (CGU), which results in a dep-node for each CGU. In subsequent
compilation sessions it then tries to mark the dep-node for a CGU as green. If
it succeeds, it knows that the corresponding object and bitcode files on disk
are still valid. If it doesn't succeed, the entire CGU has to be recompiled.

This is the same approach that is used for regular queries. The main differences
are:

 - that we cannot easily compute a fingerprint for LLVM modules (because
   they are opaque C++ objects),

 - that the logic for dealing with cached values is rather different from
   regular queries because here we have bitcode and object files instead of
   serialized Rust values in the common result cache file, and

 - the operations around LLVM are so expensive in terms of computation time and
   memory consumption that we need to have tight control over what is
   executed when and what stays in memory for how long.

The query system could probably be extended with general purpose mechanisms to
deal with all of the above but so far that seemed like more trouble than it
would save.



## Query modifiers

The query system allows for applying [modifiers][mod] to queries. These
modifiers affect certain aspects of how the system treats the query with
respect to incremental compilation:

 - `eval_always` - A query with the `eval_always` attribute is re-executed
   unconditionally during incremental compilation. I.e. the system will not
   even try to mark the query's dep-node as green. This attribute has two use
   cases:

    - `eval_always` queries can read inputs (from files, global state, etc).
      They can also produce side effects like writing to files and changing global state.

    - Some queries are very likely to be re-evaluated because their result
      depends on the entire source code. In this case `eval_always` can be used
      as an optimization because the system can skip recording dependencies in
      the first place.

 - `no_hash` - Applying `no_hash` to a query tells the system to not compute
   the fingerprint of the query's result. This has two consequences:

    - Not computing the fingerprint can save quite a bit of time because
      fingerprinting is expensive, especially for large, complex values.

    - Without the fingerprint, the system has to unconditionally assume that
      the result of the query has changed. As a consequence anything depending
      on a `no_hash` query will always be re-executed.

   Using `no_hash` for a query can make sense in two circumstances:

    - If the result of the query is very likely to change whenever one of its
      inputs changes, e.g. a function like `|a, b, c| -> (a * b * c)`. In such
      a case recomputing the query will always yield a red node if one of the
      inputs is red so we can spare us the trouble and default to red immediately.
      A counter example would be a function like `|a| -> (a == 42)` where the
      result does not change for most changes of `a`.

    - If the result of a query is a big, monolithic collection (e.g. `index_hir`)
      and there are "projection queries" reading from that collection
      (e.g. `hir_owner`). In such a case the big collection will likely fulfill the
      condition above (any changed input means recomputing the whole collection)
      and the results of the projection queries will be hashed anyway. If we also
      hashed the collection query it would mean that we effectively hash the same
      data twice: once when hashing the collection and another time when hashing all
      the projection query results. `no_hash` allows us to avoid that redundancy
      and the projection queries act as a "firewall", shielding their dependents
      from the unconditionally red `no_hash` node.

 - `cache_on_disk_if` - This attribute is what determines which query results
   are persisted in the incremental compilation query result cache. The
   attribute takes an expression that allows per query invocation
   decisions. For example, it makes no sense to store values from upstream
   crates in the cache because they are already available in the upstream
   crate's metadata.

 - `anon` - This attribute makes the system use "anonymous" dep-nodes for the
   given query. An anonymous dep-node is not identified by the corresponding
   query key, instead its ID is computed from the IDs of its dependencies. This
   allows the red-green system to do its change detection even if there is no
   query key available for a given dep-node -- something which is needed for
   handling trait selection because it is not based on queries.

[mod]: ../query.html#adding-a-new-kind-of-query


## The projection query pattern

It's interesting to note that `eval_always` and `no_hash` can be used together
in the so-called "projection query" pattern. It is often the case that there is
one query that depends on the entirety of the compiler's input (e.g. the indexed HIR)
and another query that projects individual values out of this monolithic value
(e.g. a HIR item with a certain `DefId`). These projection queries allow for
building change propagation "firewalls" because even if the result of the
monolithic query changes (which it is very likely to do) the small projections
can still mostly be marked as green.


```ignore
  +------------+
  |            |           +---------------+           +--------+
  |            | <---------| projection(x) | <---------| foo(a) |
  |            |           +---------------+           +--------+
  |            |
  | monolithic |           +---------------+           +--------+
  |   query    | <---------| projection(y) | <---------| bar(b) |
  |            |           +---------------+           +--------+
  |            |
  |            |           +---------------+           +--------+
  |            | <---------| projection(z) | <---------| baz(c) |
  |            |           +---------------+           +--------+
  +------------+
```

Let's assume that the result `monolithic_query` changes so that also the result
of `projection(x)` has changed, i.e. both their dep-nodes are being marked as
red. As a consequence `foo(a)` needs to be re-executed; but `bar(b)` and
`baz(c)` can be marked as green. However, if `foo`, `bar`, and `baz` would have
directly depended on `monolithic_query` then all of them would have had to be
re-evaluated.

This pattern works even without `eval_always` and `no_hash` but the two
modifiers can be used to avoid unnecessary overhead. If the monolithic query
is likely to change at any minor modification of the compiler's input it makes
sense to mark it as `eval_always`, thus getting rid of its dependency tracking
cost. And it always makes sense to mark the monolithic query as `no_hash`
because we have the projections to take care of keeping things green as much
as possible.


# Shortcomings of the current system

There are many things that still can be improved.

## Incrementality of on-disk data structures

The current system is not able to update on-disk caches and the dependency graph
in-place. Instead it has to rewrite each file entirely in each compilation
session. The overhead of doing so is a few percent of total compilation time.

## Unnecessary data dependencies

Data structures used as query results could be factored in a way that removes
edges from the dependency graph. Especially "span" information is very volatile,
so including it in query result will increase the chance that the result won't
be reusable. See <https://github.com/rust-lang/rust/issues/47389> for more
information.


[query-model]: ./query-evaluation-model-in-detail.html
[try_mark_green]: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_query_system/dep_graph/graph.rs.html
