# The solver

Also consider reading the documentation for [the recursive solver in chalk][chalk]
as it is very similar to this implementation and also talks about limitations of this
approach.

[chalk]: https://rust-lang.github.io/chalk/book/recursive.html

The basic structure of the solver is a pure function
`fn evaluate_goal(goal: Goal<'tcx>) -> Response`.
While the actual solver is not fully pure to deal with overflow and cycles, we are
going to defer that for now.

To deal with inference variables and to improve caching, we use
[canonicalization](./canonicalization.md).

TODO: write the remaining code for this as well.
