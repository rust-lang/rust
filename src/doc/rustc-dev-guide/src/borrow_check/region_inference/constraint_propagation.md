# Constraint propagation

The main work of the region inference is **constraint propagation**,
which is done in the [`propagate_constraints`] function.  There are
three sorts of constraints that are used in NLL, and we'll explain how
`propagate_constraints` works by "layering" those sorts of constraints
on one at a time (each of them is fairly independent from the others):

- liveness constraints (`R live at E`), which arise from liveness;
- outlives constraints (`R1: R2`), which arise from subtyping;
- [member constraints][m_c] (`member R_m of [R_c...]`), which arise from impl Trait.

[`propagate_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.propagate_constraints
[m_c]: ./member_constraints.md

In this chapter, we'll explain the "heart" of constraint propagation,
covering both liveness and outlives constraints.

## Notation and high-level concepts

Conceptually, region inference is a "fixed-point" computation. It is
given some set of constraints `{C}` and it computes a set of values
`Values: R -> {E}` that maps each region `R` to a set of elements
`{E}` (see [here][riv] for more notes on region elements):

- Initially, each region is mapped to an empty set, so `Values(R) =
  {}` for all regions `R`.
- Next, we process the constraints repeatedly until a fixed-point is reached:
  - For each constraint C:
    - Update `Values` as needed to satisfy the constraint

[riv]: ../region_inference.md#region-variables

As a simple example, if we have a liveness constraint `R live at E`,
then we can apply `Values(R) = Values(R) union {E}` to make the
constraint be satisfied. Similarly, if we have an outlives constraints
`R1: R2`, we can apply `Values(R1) = Values(R1) union Values(R2)`.
(Member constraints are more complex and we discuss them [in this section][m_c].)

In practice, however, we are a bit more clever. Instead of applying
the constraints in a loop, we can analyze the constraints and figure
out the correct order to apply them, so that we only have to apply
each constraint once in order to find the final result.

Similarly, in the implementation, the `Values` set is stored in the
`scc_values` field, but they are indexed not by a *region* but by a
*strongly connected component* (SCC). SCCs are an optimization that
avoids a lot of redundant storage and computation.  They are explained
in the section on outlives constraints.

## Liveness constraints

A **liveness constraint** arises when some variable whose type
includes a region R is live at some [point] P. This simply means that
the value of R must include the point P. Liveness constraints are
computed by the MIR type checker.

[point]: ../../appendix/glossary.md#point

A liveness constraint `R live at E` is satisfied if `E` is a member of
`Values(R)`. So to "apply" such a constraint to `Values`, we just have
to compute `Values(R) = Values(R) union {E}`.

The liveness values are computed in the type-check and passed to the
region inference upon creation in the `liveness_constraints` argument.
These are not represented as individual constraints like `R live at E`
though; instead, we store a (sparse) bitset per region variable (of
type [`LivenessValues`]). This way we only need a single bit for each
liveness constraint.

[`liveness_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.liveness_constraints
[`LivenessValues`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/values/struct.LivenessValues.html

One thing that is worth mentioning: All lifetime parameters are always
considered to be live over the entire function body. This is because
they correspond to some portion of the *caller's* execution, and that
execution clearly includes the time spent in this function, since the
caller is waiting for us to return.

## Outlives constraints

An outlives constraint `'a: 'b` indicates that the value of `'a` must
be a **superset** of the value of `'b`. That is, an outlives
constraint `R1: R2` is satisfied if `Values(R1)` is a superset of
`Values(R2)`. So to "apply" such a constraint to `Values`, we just
have to compute `Values(R1) = Values(R1) union Values(R2)`.

One observation that follows from this is that if you have `R1: R2`
and `R2: R1`, then `R1 = R2` must be true. Similarly, if you have:

```txt
R1: R2
R2: R3
R3: R4
R4: R1
```

then `R1 = R2 = R3 = R4` follows. We take advantage of this to make things
much faster, as described shortly.

In the code, the set of outlives constraints is given to the region
inference context on creation in a parameter of type
[`OutlivesConstraintSet`]. The constraint set is basically just a list of `'a:
'b` constraints.

### The outlives constraint graph and SCCs

In order to work more efficiently with outlives constraints, they are
[converted into the form of a graph][graph-fn], where the nodes of the
graph are region variables (`'a`, `'b`) and each constraint `'a: 'b`
induces an edge `'a -> 'b`. This conversion happens in the
[`RegionInferenceContext::new`] function that creates the inference
context.

[`OutlivesConstraintSet`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/constraints/struct.OutlivesConstraintSet.html
[graph-fn]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/constraints/struct.OutlivesConstraintSet.html#method.graph
[`RegionInferenceContext::new`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.new

When using a graph representation, we can detect regions that must be equal
by looking for cycles. That is, if you have a constraint like

```txt
'a: 'b
'b: 'c
'c: 'd
'd: 'a
```

then this will correspond to a cycle in the graph containing the
elements `'a...'d`.

Therefore, one of the first things that we do in propagating region
values is to compute the **strongly connected components** (SCCs) in
the constraint graph. The result is stored in the [`constraint_sccs`]
field. You can then easily find the SCC that a region `r` is a part of
by invoking `constraint_sccs.scc(r)`.

Working in terms of SCCs allows us to be more efficient: if we have a
set of regions `'a...'d` that are part of a single SCC, we don't have
to compute/store their values separately. We can just store one value
**for the SCC**, since they must all be equal.

If you look over the region inference code, you will see that a number
of fields are defined in terms of SCCs. For example, the
[`scc_values`] field stores the values of each SCC. To get the value
of a specific region `'a` then, we first figure out the SCC that the
region is a part of, and then find the value of that SCC.

[`constraint_sccs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.constraint_sccs
[`scc_values`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.scc_values

When we compute SCCs, we not only figure out which regions are a
member of each SCC, we also figure out the edges between them. So for example
consider this set of outlives constraints:

```txt
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

### Applying liveness constraints to SCCs

The liveness constraints that come in from the type-checker are
expressed in terms of regions -- that is, we have a map like
`Liveness: R -> {E}`.  But we want our final result to be expressed
in terms of SCCs -- we can integrate these liveness constraints very
easily just by taking the union:

```txt
for each region R:
  let S be the SCC that contains R
  Values(S) = Values(S) union Liveness(R)
```

In the region inferencer, this step is done in [`RegionInferenceContext::new`].

### Applying outlives constraints

Once we have computed the DAG of SCCs, we use that to structure out
entire computation. If we have an edge `S1 -> S2` between two SCCs,
that means that `Values(S1) >= Values(S2)` must hold. So, to compute
the value of `S1`, we first compute the values of each successor `S2`.
Then we simply union all of those values together. To use a
quasi-iterator-like notation:

```txt
Values(S1) =
  s1.successors()
    .map(|s2| Values(s2))
    .union()
```

In the code, this work starts in the [`propagate_constraints`]
function, which iterates over all the SCCs. For each SCC `S1`, we
compute its value by first computing the value of its
successors. Since SCCs form a DAG, we don't have to be concerned about
cycles, though we do need to keep a set around to track whether we
have already processed a given SCC or not. For each successor `S2`, once
we have computed `S2`'s value, we can union those elements into the
value for `S1`. (Although we have to be careful in this process to
properly handle [higher-ranked
placeholders](./placeholders_and_universes.html). Note that the value
for `S1` already contains the liveness constraints, since they were
added in [`RegionInferenceContext::new`].

Once that process is done, we now have the "minimal value" for `S1`,
taking into account all of the liveness and outlives
constraints. However, in order to complete the process, we must also
consider [member constraints][m_c], which are described in [a later
section][m_c].
