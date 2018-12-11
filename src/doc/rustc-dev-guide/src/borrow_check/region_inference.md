# Region inference (NLL)

The MIR-based region checking code is located in
[the `rustc_mir::borrow_check::nll` module][nll]. (NLL, of course,
stands for "non-lexical lifetimes", a term that will hopefully be
deprecated once they become the standard kind of lifetime.)

[nll]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/index.html

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
  - To do that, it first runs the [MIR type checker]. This
    is basically a normal type-checker but specialized to MIR, which
    is much simpler than full Rust, of course. Running the MIR type
    checker will however create **outlives constraints** between
    region variables (e.g., that one variable must outlive another
    one) to reflect the subtyping relationships that arise.
  - It also adds **liveness constraints** that arise from where variables
    are used.
  - After this, we create a [`RegionInferenceContext`] with the constraints we
    have computed and the inference variables we introduced and use the
    [`solve`] method to infer values for all region inference varaibles.
  - The [NLL RFC] also includes fairly thorough (and hopefully readable)
    coverage.

[fvb]: ../appendix/background.html#free-vs-bound
[`replace_regions_in_mir`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/fn.replace_regions_in_mir.html
[`compute_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/fn.compute_regions.html
[`RegionInferenceContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html
[`solve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.solve
[NLL RFC]: http://rust-lang.github.io/rfcs/2094-nll.html
[MIR type checker]: ./type_check.md

## Universal regions

The [`UnversalRegions`] type represents a collection of _universal_ regions
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

[`UniversalRegions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/universal_regions/struct.UniversalRegions.html

## Region variables

The value of a region can be thought of as a **set**. This set contains all
points in the MIR where the region is valid along with any regions that are
outlived by this region (e.g. if `'a: 'b`, then `end('b)` is in the set for
`'a`); we call the domain of this set a `RegionElement`. In the code, the value
for all regions is maintained in [the
`rustc_mir::borrow_check::nll::region_infer` module][ri]. For each region we
maintain a set storing what elements are present in its value (to make this
efficient, we give each kind of element an index, the `RegionElementIndex`, and
use sparse bitsets).

[ri]: https://github.com/rust-lang/rust/tree/master/src/librustc_mir/borrow_check/nll/region_infer/

The kinds of region elements are as follows:

- Each **location** in the MIR control-flow graph: a location is just
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
  [placeholders and universes](#placeholder).

## Constraints

Before we can infer the value of regions, we need to collect constraints on the
regions. There are two primary types of constraints.

1. Outlives constraints. These are constraints that one region outlives another
   (e.g. `'a: 'b`). Outlives constraints are generated by the [MIR type
   checker].
2. Liveness constraints. Each region needs to be live at points where it can be
   used. These constraints are collected by [`generate_constraints`].

[`generate_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/constraint_generation/fn.generate_constraints.html

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
  closures back out to the creater of the closure.

[`constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.constraints
[`liveness_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.liveness_constraints
[`universal_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.universal_regions
[`universal_region_relations`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.universal_region_relations
[`type_tests`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.type_tests
[`closure_bounds_mapping`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#structfield.closure_bounds_mapping

TODO: should we discuss any of the others fields? What about the SCCs?

Ok, now that we have constructed a `RegionInferenceContext`, we can do
inference. This is done by calling the [`solve`] method on the context. This
is where we call [`propagate_constraints`] and then check the resulting type
tests and universal regions, as discussed above.

[`propagate_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.propagate_constraints
[`check_type_tests`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.check_type_tests
[`check_universal_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/borrow_check/nll/region_infer/struct.RegionInferenceContext.html#method.check_universal_regions

## Closures

When we are checking the type tests and universal regions, we may come across a
constraint that we can't prove yet if we are in a closure body! However, the
necessary constraints may actually hold (we just don't know it yet). Thus, if
we are inside a closure, we just collect all the constraints we can't prove yet
and return them. Later, when we are borrow check the MIR node that created the
closure, we can also check that these constraints hold. At that time, if we
can't prove they hold, we report an error.

## Placeholders and universes

(This section describes ongoing work that hasn't landed yet.)

From time to time we have to reason about regions that we can't
concretely know. For example, consider this program:

```rust,ignore
// A function that needs a static reference
fn foo(x: &'static u32) { }

fn bar(f: for<'a> fn(&'a u32)) {
       // ^^^^^^^^^^^^^^^^^^^ a function that can accept **any** reference
    let x = 22;
    f(&x);
}

fn main() {
    bar(foo);
}
```

This program ought not to type-check: `foo` needs a static reference
for its argument, and `bar` wants to be given a function that that
accepts **any** reference (so it can call it with something on its
stack, for example). But *how* do we reject it and *why*?

### Subtyping and Placeholders

When we type-check `main`, and in particular the call `bar(foo)`, we
are going to wind up with a subtyping relationship like this one:

```text
fn(&'static u32) <: for<'a> fn(&'a u32)
----------------    -------------------
the type of `foo`   the type `bar` expects
```

We handle this sort of subtyping by taking the variables that are
bound in the supertype and replacing them with
[universally quantified](../appendix/background.html#quantified)
representatives, written like `!1`. We call these regions "placeholder
regions" – they represent, basically, "some unknown region".

Once we've done that replacement, we have the following relation:

```text
fn(&'static u32) <: fn(&'!1 u32)
```

The key idea here is that this unknown region `'!1` is not related to
any other regions. So if we can prove that the subtyping relationship
is true for `'!1`, then it ought to be true for any region, which is
what we wanted.

So let's work through what happens next. To check if two functions are
subtypes, we check if their arguments have the desired relationship
(fn arguments are [contravariant](../appendix/background.html#variance), so
we swap the left and right here):

```text
&'!1 u32 <: &'static u32
```

According to the basic subtyping rules for a reference, this will be
true if `'!1: 'static`. That is – if "some unknown region `!1`" lives
outlives `'static`. Now, this *might* be true – after all, `'!1`
could be `'static` – but we don't *know* that it's true. So this
should yield up an error (eventually).

### What is a universe

In the previous section, we introduced the idea of a placeholder
region, and we denoted it `!1`. We call this number `1` the **universe
index**. The idea of a "universe" is that it is a set of names that
are in scope within some type or at some point. Universes are formed
into a tree, where each child extends its parents with some new names.
So the **root universe** conceptually contains global names, such as
the the lifetime `'static` or the type `i32`. In the compiler, we also
put generic type parameters into this root universe (in this sense,
there is not just one root universe, but one per item). So consider
this function `bar`:

```rust,ignore
struct Foo { }

fn bar<'a, T>(t: &'a T) {
    ...
}
```

Here, the root universe would consist of the lifetimes `'static` and
`'a`.  In fact, although we're focused on lifetimes here, we can apply
the same concept to types, in which case the types `Foo` and `T` would
be in the root universe (along with other global types, like `i32`).
Basically, the root universe contains all the names that
[appear free](../appendix/background.html#free-vs-bound) in the body of `bar`.

Now let's extend `bar` a bit by adding a variable `x`:

```rust,ignore
fn bar<'a, T>(t: &'a T) {
    let x: for<'b> fn(&'b u32) = ...;
}
```

Here, the name `'b` is not part of the root universe. Instead, when we
"enter" into this `for<'b>` (e.g., by replacing it with a placeholder), we will create
a child universe of the root, let's call it U1:

```text
U0 (root universe)
│
└─ U1 (child universe)
```

The idea is that this child universe U1 extends the root universe U0
with a new name, which we are identifying by its universe number:
`!1`.

Now let's extend `bar` a bit by adding one more variable, `y`:

```rust,ignore
fn bar<'a, T>(t: &'a T) {
    let x: for<'b> fn(&'b u32) = ...;
    let y: for<'c> fn(&'b u32) = ...;
}
```

When we enter *this* type, we will again create a new universe, which
we'll call `U2`. Its parent will be the root universe, and U1 will be
its sibling:

```text
U0 (root universe)
│
├─ U1 (child universe)
│
└─ U2 (child universe)
```

This implies that, while in U2, we can name things from U0 or U2, but
not U1.

**Giving existential variables a universe.** Now that we have this
notion of universes, we can use it to extend our type-checker and
things to prevent illegal names from leaking out. The idea is that we
give each inference (existential) variable – whether it be a type or
a lifetime – a universe. That variable's value can then only
reference names visible from that universe. So for example is a
lifetime variable is created in U0, then it cannot be assigned a value
of `!1` or `!2`, because those names are not visible from the universe
U0.

**Representing universes with just a counter.** You might be surprised
to see that the compiler doesn't keep track of a full tree of
universes. Instead, it just keeps a counter – and, to determine if
one universe can see another one, it just checks if the index is
greater. For example, U2 can see U0 because 2 >= 0. But U0 cannot see
U2, because 0 >= 2 is false.

How can we get away with this? Doesn't this mean that we would allow
U2 to also see U1? The answer is that, yes, we would, **if that
question ever arose**.  But because of the structure of our type
checker etc, there is no way for that to happen. In order for
something happening in the universe U1 to "communicate" with something
happening in U2, they would have to have a shared inference variable X
in common. And because everything in U1 is scoped to just U1 and its
children, that inference variable X would have to be in U0. And since
X is in U0, it cannot name anything from U1 (or U2). This is perhaps easiest
to see by using a kind of generic "logic" example:

```text
exists<X> {
   forall<Y> { ... /* Y is in U1 ... */ }
   forall<Z> { ... /* Z is in U2 ... */ }
}
```

Here, the only way for the two foralls to interact would be through X,
but neither Y nor Z are in scope when X is declared, so its value
cannot reference either of them.

### Universes and placeholder region elements

But where does that error come from?  The way it happens is like this.
When we are constructing the region inference context, we can tell
from the type inference context how many placeholder variables exist
(the `InferCtxt` has an internal counter). For each of those, we
create a corresponding universal region variable `!n` and a "region
element" `placeholder(n)`. This corresponds to "some unknown set of other
elements". The value of `!n` is `{placeholder(n)}`.

At the same time, we also give each existential variable a
**universe** (also taken from the `InferCtxt`). This universe
determines which placeholder elements may appear in its value: For
example, a variable in universe U3 may name `placeholder(1)`, `placeholder(2)`, and
`placeholder(3)`, but not `placeholder(4)`. Note that the universe of an inference
variable controls what region elements **can** appear in its value; it
does not say region elements **will** appear.

### Placeholders and outlives constraints

In the region inference engine, outlives constraints have the form:

```text
V1: V2 @ P
```

where `V1` and `V2` are region indices, and hence map to some region
variable (which may be universally or existentially quantified). The
`P` here is a "point" in the control-flow graph; it's not important
for this section. This variable will have a universe, so let's call
those universes `U(V1)` and `U(V2)` respectively. (Actually, the only
one we are going to care about is `U(V1)`.)

When we encounter this constraint, the ordinary procedure is to start
a DFS from `P`. We keep walking so long as the nodes we are walking
are present in `value(V2)` and we add those nodes to `value(V1)`. If
we reach a return point, we add in any `end(X)` elements. That part
remains unchanged.

But then *after that* we want to iterate over the placeholder `placeholder(x)`
elements in V2 (each of those must be visible to `U(V2)`, but we
should be able to just assume that is true, we don't have to check
it). We have to ensure that `value(V1)` outlives each of those
placeholder elements.

Now there are two ways that could happen. First, if `U(V1)` can see
the universe `x` (i.e., `x <= U(V1)`), then we can just add `placeholder(x)`
to `value(V1)` and be done. But if not, then we have to approximate:
we may not know what set of elements `placeholder(x)` represents, but we
should be able to compute some sort of **upper bound** B for it –
some region B that outlives `placeholder(x)`. For now, we'll just use
`'static` for that (since it outlives everything) – in the future, we
can sometimes be smarter here (and in fact we have code for doing this
already in other contexts). Moreover, since `'static` is in the root
universe U0, we know that all variables can see it – so basically if
we find that `value(V2)` contains `placeholder(x)` for some universe `x`
that `V1` can't see, then we force `V1` to `'static`.

### Extending the "universal regions" check

After all constraints have been propagated, the NLL region inference
has one final check, where it goes over the values that wound up being
computed for each universal region and checks that they did not get
'too large'. In our case, we will go through each placeholder region
and check that it contains *only* the `placeholder(u)` element it is known to
outlive. (Later, we might be able to know that there are relationships
between two placeholder regions and take those into account, as we do
for universal regions from the fn signature.)

Put another way, the "universal regions" check can be considered to be
checking constraints like:

```text
{placeholder(1)}: V1
```

where `{placeholder(1)}` is like a constant set, and V1 is the variable we
made to represent the `!1` region.

## Back to our example

OK, so far so good. Now let's walk through what would happen with our
first example:

```text
fn(&'static u32) <: fn(&'!1 u32) @ P  // this point P is not imp't here
```

The region inference engine will create a region element domain like this:

```text
{ CFG; end('static); placeholder(1) }
    ---  ------------  ------- from the universe `!1`
    |    'static is always in scope
    all points in the CFG; not especially relevant here
```

It will always create two universal variables, one representing
`'static` and one representing `'!1`. Let's call them Vs and V1. They
will have initial values like so:

```text
Vs = { CFG; end('static) } // it is in U0, so can't name anything else
V1 = { placeholder(1) }
```

From the subtyping constraint above, we would have an outlives constraint like

```text
'!1: 'static @ P
```

To process this, we would grow the value of V1 to include all of Vs:

```text
Vs = { CFG; end('static) }
V1 = { CFG; end('static), placeholder(1) }
```

At that point, constraint propagation is complete, because all the
outlives relationships are satisfied. Then we would go to the "check
universal regions" portion of the code, which would test that no
universal region grew too large.

In this case, `V1` *did* grow too large – it is not known to outlive
`end('static)`, nor any of the CFG – so we would report an error.

## Another example

What about this subtyping relationship?

```text
for<'a> fn(&'a u32, &'a u32)
    <:
for<'b, 'c> fn(&'b u32, &'c u32)
```

Here we would replace the bound region in the supertype with a placeholder, as before, yielding:

```text
for<'a> fn(&'a u32, &'a u32)
    <:
fn(&'!1 u32, &'!2 u32)
```

then we instantiate the variable on the left-hand side with an
existential in universe U2, yielding the following (`?n` is a notation
for an existential variable):

```text
fn(&'?3 u32, &'?3 u32)
    <:
fn(&'!1 u32, &'!2 u32)
```

Then we break this down further:

```text
&'!1 u32 <: &'?3 u32
&'!2 u32 <: &'?3 u32
```

and even further, yield up our region constraints:

```text
'!1: '?3
'!2: '?3
```

Note that, in this case, both `'!1` and `'!2` have to outlive the
variable `'?3`, but the variable `'?3` is not forced to outlive
anything else. Therefore, it simply starts and ends as the empty set
of elements, and hence the type-check succeeds here.

(This should surprise you a little. It surprised me when I first realized it.
We are saying that if we are a fn that **needs both of its arguments to have
the same region**, we can accept being called with **arguments with two
distinct regions**. That seems intuitively unsound. But in fact, it's fine, as
I tried to explain in [this issue][ohdeargoditsallbroken] on the Rust issue
tracker long ago.  The reason is that even if we get called with arguments of
two distinct lifetimes, those two lifetimes have some intersection (the call
itself), and that intersection can be our value of `'a` that we use as the
common lifetime of our arguments. -nmatsakis)

[ohdeargoditsallbroken]: https://github.com/rust-lang/rust/issues/32330#issuecomment-202536977

## Final example

Let's look at one last example. We'll extend the previous one to have
a return type:

```text
for<'a> fn(&'a u32, &'a u32) -> &'a u32
    <:
for<'b, 'c> fn(&'b u32, &'c u32) -> &'b u32
```

Despite seeming very similar to the previous example, this case is going to get
an error. That's good: the problem is that we've gone from a fn that promises
to return one of its two arguments, to a fn that is promising to return the
first one. That is unsound. Let's see how it plays out.

First, we replace the bound region in the supertype with a placeholder:

```text
for<'a> fn(&'a u32, &'a u32) -> &'a u32
    <:
fn(&'!1 u32, &'!2 u32) -> &'!1 u32
```

Then we instantiate the subtype with existentials (in U2):

```text
fn(&'?3 u32, &'?3 u32) -> &'?3 u32
    <:
fn(&'!1 u32, &'!2 u32) -> &'!1 u32
```

And now we create the subtyping relationships:

```text
&'!1 u32 <: &'?3 u32 // arg 1
&'!2 u32 <: &'?3 u32 // arg 2
&'?3 u32 <: &'!1 u32 // return type
```

And finally the outlives relationships. Here, let V1, V2, and V3 be the
variables we assign to `!1`, `!2`, and `?3` respectively:

```text
V1: V3
V2: V3
V3: V1
```

Those variables will have these initial values:

```text
V1 in U1 = {placeholder(1)}
V2 in U2 = {placeholder(2)}
V3 in U2 = {}
```

Now because of the `V3: V1` constraint, we have to add `placeholder(1)` into `V3` (and
indeed it is visible from `V3`), so we get:

```text
V3 in U2 = {placeholder(1)}
```

then we have this constraint `V2: V3`, so we wind up having to enlarge
`V2` to include `placeholder(1)` (which it can also see):

```text
V2 in U2 = {placeholder(1), placeholder(2)}
```

Now constraint propagation is done, but when we check the outlives
relationships, we find that `V2` includes this new element `placeholder(1)`,
so we report an error.

## Borrow Checker Errors

TODO: we should discuss how to generate errors from the results of these analyses.
