# Universal regions

"Universal regions" is the name that the code uses to refer to "named
lifetimes" -- e.g., lifetime parameters and `'static`. The name
derives from the fact that such lifetimes are "universally quantified"
(i.e., we must make sure the code is true for all values of those
lifetimes). It is worth spending a bit of discussing how lifetime
parameters are handled during region inference. Consider this example:

```rust,ignore
fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> &'b u32 {
  x
}
```

This example is intended not to compile, because we are returning `x`,
which has type `&'a u32`, but our signature promises that we will
return a `&'b u32` value. But how are lifetimes like `'a` and `'b`
integrated into region inference, and how this error wind up being
detected?

## Universal regions and their relationships to one another

Early on in region inference, one of the first things we do is to
construct a [`UniversalRegions`] struct. This struct tracks the
various universal regions in scope on a particular function.  We also
create a [`UniversalRegionRelations`] struct, which tracks their
relationships to one another. So if you have e.g. `where 'a: 'b`, then
the [`UniversalRegionRelations`] struct would track that `'a: 'b` is
known to hold (which could be tested with the [`outlives`] function).

[`UniversalRegions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/universal_regions/struct.UniversalRegions.html
[`UniversalRegionRelations`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/type_check/free_region_relations/struct.UniversalRegionRelations.html
[`outlives`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/type_check/free_region_relations/struct.UniversalRegionRelations.html#method.outlives

## Everything is a region variable

One important aspect of how NLL region inference works is that **all
lifetimes** are represented as numbered variables. This means that the
only variant of [`region_kind::RegionKind`] that we use is the [`ReVar`]
variant. These region variables are broken into two major categories,
based on their index:

[`region_kind::RegionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/region_kind/enum.RegionKind.html
[`ReVar`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/region_kind/enum.RegionKind.html#variant.ReVar

- 0..N: universal regions -- the ones we are discussing here. In this
  case, the code must be correct with respect to any value of those
  variables that meets the declared relationships.
- N..M: existential regions -- inference variables where the region
  inferencer is tasked with finding *some* suitable value.

In fact, the universal regions can be further subdivided based on
where they were brought into scope (see the [`RegionClassification`]
type). These subdivisions are not important for the topics discussed
here, but become important when we consider [closure constraint
propagation](./closure_constraints.html), so we discuss them there.

[`RegionClassification`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/universal_regions/enum.RegionClassification.html#variant.Local

## Universal lifetimes as the elements of a region's value

As noted previously, the value that we infer for each region is a set
`{E}`. The elements of this set can be points in the control-flow
graph, but they can also be an element `end('a)` corresponding to each
universal lifetime `'a`. If the value for some region `R0` includes
`end('a`), then this implies that `R0` must extend until the end of `'a`
in the caller.

## The "value" of a universal region

During region inference, we compute a value for each universal region
in the same way as we compute values for other regions. This value
represents, effectively, the **lower bound** on that universal region
-- the things that it must outlive. We now describe how we use this
value to check for errors.

## Liveness and universal regions

All universal regions have an initial liveness constraint that
includes the entire function body. This is because lifetime parameters
are defined in the caller and must include the entirety of the
function call that invokes this particular function. In addition, each
universal region `'a` includes itself (that is, `end('a)`) in its
liveness constraint (i.e., `'a` must extend until the end of
itself). In the code, these liveness constraints are setup in
[`init_free_and_bound_regions`].

[`init_free_and_bound_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.init_free_and_bound_regions

## Propagating outlives constraints for universal regions

So, consider the first example of this section:

```rust,ignore
fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> &'b u32 {
  x
}
```

Here, returning `x` requires that `&'a u32 <: &'b u32`, which gives
rise to an outlives constraint `'a: 'b`. Combined with our default liveness
constraints we get:

```txt
'a live at {B, end('a)} // B represents the "function body"
'b live at {B, end('b)}
'a: 'b
```

When we process the `'a: 'b` constraint, therefore, we will add
`end('b)` into the value for `'a`, resulting in a final value of `{B,
end('a), end('b)}`.

## Detecting errors

Once we have finished constraint propagation, we then enforce a
constraint that if some universal region `'a` includes an element
`end('b)`, then `'a: 'b` must be declared in the function's bounds. If
not, as in our example, that is an error. This check is done in the
[`check_universal_regions`] function, which simply iterates over all
universal regions, inspects their final value, and tests against the
declared [`UniversalRegionRelations`].

[`check_universal_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.check_universal_regions
