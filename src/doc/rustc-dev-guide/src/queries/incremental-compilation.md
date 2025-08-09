# Incremental compilation

The incremental compilation scheme is, in essence, a surprisingly
simple extension to the overall query system. We'll start by describing
a slightly simplified variant of the real thing – the "basic algorithm" –
and then describe some possible improvements.

## The basic algorithm

The basic algorithm is
called the **red-green** algorithm[^salsa]. The high-level idea is
that, after each run of the compiler, we will save the results of all
the queries that we do, as well as the **query DAG**. The
**query DAG** is a [DAG] that indexes which queries executed which
other queries. So, for example, there would be an [edge] from a query Q1
to another query Q2 if computing Q1 required computing Q2 (note that
because queries cannot depend on themselves, this results in a DAG and
not a general graph).

[DAG]: https://en.wikipedia.org/wiki/Directed_acyclic_graph

> **NOTE**: You might think of a query as simply the definition of a query.
> A thing that you can invoke, a bit like a function, 
> and which either returns a cached result or actually executes the code.
> 
> If that's the way you think about queries,
> it's good to know that in the following text, queries will be said to have colours. 
> Keep in mind though, that here the word query also refers to a certain invocation of 
> the query for a certain input. As you will read later, queries are fingerprinted based 
> on their arguments. The result of a query might change when we give it one argument 
> and be coloured red, while it stays the same for another argument and is thus green.
> 
> In short, the word query is here not just used to mean the definition of a query, 
> but also for a specific instance of that query with given arguments.

On the next run of the compiler, then, we can sometimes reuse these
query results to avoid re-executing a query. We do this by assigning
every query a **color**:

- If a query is colored **red**, that means that its result during
  this compilation has **changed** from the previous compilation.
- If a query is colored **green**, that means that its result is
  the **same** as the previous compilation.

There are two key insights here:

- First, if all the inputs to query Q are colored green, then the
  query Q **must** result in the same value as last time and hence
  need not be re-executed (or else the compiler is not deterministic).
- Second, even if some inputs to a query changes, it may be that it
  **still** produces the same result as the previous compilation. In
  particular, the query may only use part of its input.
  - Therefore, after executing a query, we always check whether it
    produced the same result as the previous time. **If it did,** we
    can still mark the query as green, and hence avoid re-executing
    dependent queries.

### The try-mark-green algorithm

At the core of incremental compilation is an algorithm called
"try-mark-green". It has the job of determining the color of a given
query Q (which must not have yet been executed). In cases where Q has
red inputs, determining Q's color may involve re-executing Q so that
we can compare its output, but if all of Q's inputs are green, then we
can conclude that Q must be green without re-executing it or inspecting
its value at all. In the compiler, this allows us to avoid
deserializing the result from disk when we don't need it, and in fact
enables us to sometimes skip *serializing* the result as well
(see the refinements section below).

Try-mark-green works as follows:

- First check if the query Q was executed during the previous compilation.
  - If not, we can just re-execute the query as normal, and assign it the
    color of red.
- If yes, then load the 'dependent queries' of Q.
- If there is a saved result, then we load the `reads(Q)` vector from the
  query DAG. The "reads" is the set of queries that Q executed during
  its execution.
  - For each query R in `reads(Q)`, we recursively demand the color
    of R using try-mark-green.
    - Note: it is important that we visit each node in `reads(Q)` in same order
      as they occurred in the original compilation. See [the section on the
      query DAG below](#dag).
    - If **any** of the nodes in `reads(Q)` wind up colored **red**, then Q is
      dirty.
      - We re-execute Q and compare the hash of its result to the hash of the
        result from the previous compilation.
      - If the hash has not changed, we can mark Q as **green** and return.
    - Otherwise, **all** of the nodes in `reads(Q)` must be **green**. In that
      case, we can color Q as **green** and return.

<a id="dag"></a>

### The query DAG

The query DAG code is stored in
[`compiler/rustc_middle/src/dep_graph`][dep_graph]. Construction of the DAG is done
by instrumenting the query execution.

One key point is that the query DAG also tracks ordering; that is, for
each query Q, we not only track the queries that Q reads, we track the
**order** in which they were read.  This allows try-mark-green to walk
those queries back in the same order. This is important because once a
subquery comes back as red, we can no longer be sure that Q will continue
along the same path as before. That is, imagine a query like this:

```rust,ignore
fn main_query(tcx) {
    if tcx.subquery1() {
        tcx.subquery2()
    } else {
        tcx.subquery3()
    }
}
```

Now imagine that in the first compilation, `main_query` starts by
executing `subquery1`, and this returns true. In that case, the next
query `main_query` executes will be `subquery2`, and `subquery3` will
not be executed at all.

But now imagine that in the **next** compilation, the input has
changed such that `subquery1` returns **false**. In this case, `subquery2`
would never execute. If try-mark-green were to visit `reads(main_query)` out
of order, however, it might visit `subquery2` before `subquery1`, and hence
execute it.
This can lead to ICEs and other problems in the compiler.

[dep_graph]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/dep_graph/index.html

## Improvements to the basic algorithm

In the description of the basic algorithm, we said that at the end of
compilation we would save the results of all the queries that were
performed.  In practice, this can be quite wasteful – many of those
results are very cheap to recompute, and serializing and deserializing
them is not a particular win. In practice, what we would do is to save
**the hashes** of all the subqueries that we performed. Then, in select cases,
we **also** save the results.

This is why the incremental algorithm separates computing the
**color** of a node, which often does not require its value, from
computing the **result** of a node. Computing the result is done via a simple
algorithm like so:

- Check if a saved result for Q is available. If so, compute the color of Q.
  If Q is green, deserialize and return the saved result.
- Otherwise, execute Q.
  - We can then compare the hash of the result and color Q as green if
    it did not change.

## Resources
The initial design document can be found [here][initial-design], which expands
on the memoization details, provides more high-level overview and motivation
for this system.

# Footnotes

[^salsa]: I have long wanted to rename it to the Salsa algorithm, but it never caught on. -@nikomatsakis

[edge]: https://en.wikipedia.org/wiki/Glossary_of_graph_theory_terms#edge
[initial-design]: https://github.com/nikomatsakis/rustc-on-demand-incremental-design-doc/blob/master/0000-rustc-on-demand-and-incremental.md
