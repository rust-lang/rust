# Specialization

**TODO**: where does Chalk fit in? Should we mention/discuss it here?

Defined in the `specialize` module.

The basic strategy is to build up a *specialization graph* during
coherence checking (coherence checking looks for [overlapping impls](../coherence.md)). 
Insertion into the graph locates the right place
to put an impl in the specialization hierarchy; if there is no right
place (due to partial overlap but no containment), you get an overlap
error. Specialization is consulted when selecting an impl (of course),
and the graph is consulted when propagating defaults down the
specialization hierarchy.

You might expect that the specialization graph would be used during
selection – i.e. when actually performing specialization. This is
not done for two reasons:

- It's merely an optimization: given a set of candidates that apply,
  we can determine the most specialized one by comparing them directly
  for specialization, rather than consulting the graph. Given that we
  also cache the results of selection, the benefit of this
  optimization is questionable.

- To build the specialization graph in the first place, we need to use
  selection (because we need to determine whether one impl specializes
  another). Dealing with this reentrancy would require some additional
  mode switch for selection. Given that there seems to be no strong
  reason to use the graph anyway, we stick with a simpler approach in
  selection, and use the graph only for propagating default
  implementations.

Trait impl selection can succeed even when multiple impls can apply,
as long as they are part of the same specialization family. In that
case, it returns a *single* impl on success – this is the most
specialized impl *known* to apply. However, if there are any inference
variables in play, the returned impl may not be the actual impl we
will use at codegen time. Thus, we take special care to avoid projecting
associated types unless either (1) the associated type does not use
`default` and thus cannot be overridden or (2) all input types are
known concretely.

## Additional Resources

[This talk][talk] by @sunjay may be useful. Keep in mind that the talk only
gives a broad overview of the problem and the solution (it was presented about
halfway through @sunjay's work). Also, it was given in June 2018, and some
things may have changed by the time you watch it.

[talk]: https://www.youtube.com/watch?v=rZqS4bLPL24
