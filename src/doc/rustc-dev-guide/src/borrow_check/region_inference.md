# Region inference (NLL)

The MIR-based region checking code is located in [the `rustc_mir::borrow_check`
module][nll].

[nll]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/index.html

The MIR-based region analysis consists of two major functions:

- [`replace_regions_in_mir`], invoked first, has two jobs:
  - First, it finds the set of regions that appear within the
    signature of the function (e.g., `'a` in `fn foo<'a>(&'a u32) {
    ... }`). These are called the "universal" or "free" regions – in
    particular, they are the regions that [appear free][fvb] in the
    function body.
  - Second, it replaces all the regions from the function body with
    fresh inference variables. This is because (presently) those
    regions are the results of lexical region inference and hence are
    not of much interest. The intention is that – eventually – they
    will be "erased regions" (i.e., no information at all), since we
    won't be doing lexical region inference at all.
- [`compute_regions`], invoked second: this is given as argument the
  results of move analysis. It has the job of computing values for all
  the inference variables that `replace_regions_in_mir` introduced.
  - To do that, it first runs the [MIR type checker]. This is
    basically a normal type-checker but specialized to MIR, which is
    much simpler than full Rust, of course. Running the MIR type
    checker will however create various [constraints][cp] between region
    variables, indicating their potential values and relationships to
    one another.
  - After this, we perform [constraint propagation][cp] by creating a
    [`RegionInferenceContext`] and invoking its [`solve`]
    method.
  - The [NLL RFC] also includes fairly thorough (and hopefully readable)
    coverage.

[cp]: ./region_inference/constraint_propagation.md
[fvb]: ../appendix/background.md#free-vs-bound
[`replace_regions_in_mir`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/nll/fn.replace_regions_in_mir.html
[`compute_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/nll/fn.compute_regions.html
[`RegionInferenceContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html
[`solve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.solve
[NLL RFC]: https://rust-lang.github.io/rfcs/2094-nll.html
[MIR type checker]: ./type_check.md

## Universal regions

The [`UniversalRegions`] type represents a collection of _universal_ regions
corresponding to some MIR `DefId`. It is constructed in
[`replace_regions_in_mir`] when we replace all regions with fresh inference
variables. [`UniversalRegions`] contains indices for all the free regions in
the given MIR along with any relationships that are _known_ to hold between
them (e.g. implied bounds, where clauses, etc.).

For example, given the MIR for the following function:

```rust
fn foo<'a>(x: &'a u32) {
    // ...
}
```

we would create a universal region for `'a` and one for `'static`. There may
also be some complications for handling closures, but we will ignore those for
the moment.

TODO: write about _how_ these regions are computed.

[`UniversalRegions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/universal_regions/struct.UniversalRegions.html

<a id="region-variables"></a>

## Region variables

The value of a region can be thought of as a **set**. This set contains all
points in the MIR where the region is valid along with any regions that are
outlived by this region (e.g. if `'a: 'b`, then `end('b)` is in the set for
`'a`); we call the domain of this set a `RegionElement`. In the code, the value
for all regions is maintained in [the `rustc_borrowck::region_infer` module][ri].
For each region we maintain a set storing what elements are present in its value (to make this
efficient, we give each kind of element an index, the `RegionElementIndex`, and
use sparse bitsets).

[ri]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_borrowck/src/region_infer

The kinds of region elements are as follows:

- Each **[`location`]** in the MIR control-flow graph: a location is just
  the pair of a basic block and an index. This identifies the point
  **on entry** to the statement with that index (or the terminator, if
  the index is equal to `statements.len()`).
- There is an element `end('a)` for each universal region `'a`,
  corresponding to some portion of the caller's (or caller's caller,
  etc) control-flow graph.
- Similarly, there is an element denoted `end('static)` corresponding
  to the remainder of program execution after this function returns.
- There is an element `!1` for each placeholder region `!1`. This
  corresponds (intuitively) to some unknown set of other elements –
  for details on placeholders, see the section
  [placeholders and universes](region_inference/placeholders_and_universes.md).

## Constraints

Before we can infer the value of regions, we need to collect
constraints on the regions. The full set of constraints is described
in [the section on constraint propagation][cp], but the two most
common sorts of constraints are:

1. Outlives constraints. These are constraints that one region outlives another
   (e.g. `'a: 'b`). Outlives constraints are generated by the [MIR type
   checker].
2. Liveness constraints. Each region needs to be live at points where it can be
   used.

## Inference Overview

So how do we compute the contents of a region? This process is called _region
inference_. The high-level idea is pretty simple, but there are some details we
need to take care of.

Here is the high-level idea: we start off each region with the MIR locations we
know must be in it from the liveness constraints. From there, we use all of the
outlives constraints computed from the type checker to _propagate_ the
constraints: for each region `'a`, if `'a: 'b`, then we add all elements of
`'b` to `'a`, including `end('b)`. This all happens in
[`propagate_constraints`].

Then, we will check for errors. We first check that type tests are satisfied by
calling [`check_type_tests`]. This checks constraints like `T: 'a`. Second, we
check that universal regions are not "too big". This is done by calling
[`check_universal_regions`]. This checks that for each region `'a` if `'a`
contains the element `end('b)`, then we must already know that `'a: 'b` holds
(e.g. from a where clause). If we don't already know this, that is an error...
well, almost. There is some special handling for closures that we will discuss
later.

### Example

Consider the following example:

```rust,ignore
fn foo<'a, 'b>(x: &'a usize) -> &'b usize {
    x
}
```

Clearly, this should not compile because we don't know if `'a` outlives `'b`
(if it doesn't then the return value could be a dangling reference).

Let's back up a bit. We need to introduce some free inference variables (as is
done in [`replace_regions_in_mir`]). This example doesn't use the exact regions
produced, but it (hopefully) is enough to get the idea across.

```rust,ignore
fn foo<'a, 'b>(x: &'a /* '#1 */ usize) -> &'b /* '#3 */ usize {
    x // '#2, location L1
}
```

Some notation: `'#1`, `'#3`, and `'#2` represent the universal regions for the
argument, return value, and the expression `x`, respectively. Additionally, I
will call the location of the expression `x` `L1`.

So now we can use the liveness constraints to get the following starting points:

Region  | Contents
--------|----------
'#1     |
'#2     | `L1`
'#3     | `L1`

Now we use the outlives constraints to expand each region. Specifically, we
know that `'#2: '#3` ...

Region  | Contents
--------|----------
'#1     | `L1`
'#2     | `L1, end('#3) // add contents of '#3 and end('#3)`
'#3     | `L1`

... and `'#1: '#2`, so ...

Region  | Contents
--------|----------
'#1     | `L1, end('#2), end('#3) // add contents of '#2 and end('#2)`
'#2     | `L1, end('#3)`
'#3     | `L1`

Now, we need to check that no regions were too big (we don't have any type
tests to check in this case). Notice that `'#1` now contains `end('#3)`, but
we have no `where` clause or implied bound to say that `'a: 'b`... that's an
error!

### Some details

The [`RegionInferenceContext`] type contains all of the information needed to
do inference, including the universal regions from [`replace_regions_in_mir`] and
the constraints computed for each region. It is constructed just after we
compute the liveness constraints.

Here are some of the fields of the struct:

- [`constraints`]: contains all the outlives constraints.
- [`liveness_constraints`]: contains all the liveness constraints.
- [`universal_regions`]: contains the `UniversalRegions` returned by
  [`replace_regions_in_mir`].
- [`universal_region_relations`]: contains relations known to be true about
  universal regions. For example, if we have a where clause that `'a: 'b`, that
  relation is assumed to be true while borrow checking the implementation (it
  is checked at the caller), so `universal_region_relations` would contain `'a:
  'b`.
- [`type_tests`]: contains some constraints on types that we must check after
  inference (e.g. `T: 'a`).
- [`closure_bounds_mapping`]: used for propagating region constraints from
  closures back out to the creator of the closure.

[`constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.constraints
[`liveness_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.liveness_constraints
[`location`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Location.html
[`universal_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.universal_regions
[`universal_region_relations`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.universal_region_relations
[`type_tests`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.type_tests
[`closure_bounds_mapping`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#structfield.closure_bounds_mapping

TODO: should we discuss any of the others fields? What about the SCCs?

Ok, now that we have constructed a `RegionInferenceContext`, we can do
inference. This is done by calling the [`solve`] method on the context. This
is where we call [`propagate_constraints`] and then check the resulting type
tests and universal regions, as discussed above.

[`propagate_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.propagate_constraints
[`check_type_tests`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.check_type_tests
[`check_universal_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/region_infer/struct.RegionInferenceContext.html#method.check_universal_regions
