# Constraint propagation

The main work of the region inference is **constraint
propagation**. This means processing the set of constraints to compute
the final values for all the region variables.

## Kinds of constraints

Each kind of constraint is handled somewhat differently by the region inferencer.

### Liveness constraints

A **liveness constraint** arises when some variable whose type
includes a region R is live at some point P. This simply means that
the value of R must include the point P. Liveness constraints are
computed by the MIR type checker.

We represent them by keeping a (sparse) bitset for each region
variable, which is the field [`liveness_constraints`], of type
[`LivenessValues`]

[`liveness_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.liveness_constraints
[`LivenessValues`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/values/struct.LivenessValues.html

### Outlives constraints

An outlives constraint `'a: 'b` indicates that the value of `'a` must
be a **superset** of the value of `'b`. On creation, we are given a
set of outlives constraints in the form of a
[`ConstraintSet`]. However, to work more efficiently with outlives
constraints, they are [converted into the form of a graph][graph-fn],
where the nodes of the graph are region variables (`'a`, `'b`) and
each constraint `'a: 'b` induces an edge `'a -> 'b`. This conversion
happens in the [`RegionInferenceContext::new`] function that creates
the inference context.

[`ConstraintSet`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/constraints/struct.ConstraintSet.html
[graph-fn]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/constraints/struct.ConstraintSet.html#method.graph
[`RegionInferenceContext::new`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.new

### Member constraints

A member constraint `'m member of ['c_1..'c_N]` expresses that the
region `'m` must be *equal* to some **choice regions** `'c_i` (for
some `i`). These constraints cannot be expressed by users, but they arise
from `impl Trait` due to its lifetime capture rules. Consinder a function
such as the following:

```rust
fn make(a: &'a u32, b: &'b u32) -> impl Trait<'a, 'b> { .. }
```

Here, the true return type (often called the "hidden type") is only
permitted to capture the lifeimes `'a` or `'b`. You can kind of see
this more clearly by desugaring that `impl Trait` return type into its
more explicit form:

```rust
type MakeReturn<'x, 'y> = impl Trait<'x, 'y>;
fn make(a: &'a u32, b: &'b u32) -> MakeReturn<'a, 'b> { .. }
```

Here, the idea is that the hidden type must be some type that could
have been written in place of the `impl Trait<'x, 'y>` -- but clearly
such a type can only reference the regions `'x` or `'y` (or
`'static`!), as those are the only names in scope. This limitation is
then translated into a restriction to only access `'a` or `'b` because
we are returning `MakeReturn<'a, 'b>`, where `'x` and `'y` have been
replaced with `'a` and `'b` respectively.

## SCCs in the outlives constraint graph

The most common sort of constraint in practice are outlives
constraints like `'a: 'b`. Such a cosntraint means that `'a` is a
superset of `'b`.  So what happens if we have two regions `'a` and `'b`
that mutually outlive one another, like so?

```
'a: 'b
'b: 'a
```

In this case, we can conclude that `'a` and `'b` must be equal
sets. In fact, it doesn't have to be just two regions. We could create
an extended "chain" of outlives constraints:

```
'a: 'b
'b: 'c
'c: 'd
'd: 'a
```

Here, we know that `'a..'d` are all equal to one another.

As mentioned above, an outlives constraint like `'a: 'b` can be viewed
as an edge in a graph `'a -> 'b`. Cycles in this graph indicate regions
that mutually outlive one another and hence must be equal. 

Therefore, one of the first things that we do in propagating region
values is to compute the **strongly connected components** (SCCs) in
the constraint graph. The result is stored in the [`constraint_sccs`]
field. You can then easily find the SCC that a region `r` is a part of
by invoking `constraint_sccs.scc(r)`.

Working in terms of SCCs allows us to be more efficient: if we have a
set of regions `'a...'d` that are part of a single SCC, we don't have
to compute/store their values separarely. We can just store one value
**for the SCC**, since they must all be equal.

If you look over the region inference code, you will see that a number
of fields are defined in terms of SCCs. For example, the
[`scc_values`] field stores the values of each SCC. To get the value
of a specific region `'a` then, we first figure out the SCC that the
region is a part of, and then find the value of that SCC.

[`constraint_sccs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.constraint_sccs
[`scc_values`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.scc_values

When we compute SCCs, we not only figure out which regions are a
member of each SCC, we also figure out the edges between them. So for example
consider this set of outlives constraints:

```
'a: 'b
'b: 'a

'a: 'c

'c: 'd
'd: 'c
```

Here we have two SCCs: S0 contains `'a` and `'b`, and S1 contains `'c`
and `'d`.  But these SCCs are not independent: because `'a: 'c`, that
means that `S0: S1` as well. That is -- the value of `S0` must be a
superset of the value of `S1`. One crucial thing is that this graph of
SCCs is always a DAG -- that is, it never has cycles. This is because
all the cycles have been removed to form the SCCs themselves.

## How constraint propagation works

The main work of constraint propagation is done in the
`propagation_constraints` function.

[`propagate_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.propagate_constraints
