# MIR-based region checking (NLL)

The MIR-based region checking code is located in
[the `rustc_mir::borrow_check::nll` module][nll]. (NLL, of course,
stands for "non-lexical lifetimes", a term that will hopefully be
deprecated once they become the standard kind of lifetime.)

[nll]: https://github.com/rust-lang/rust/tree/master/src/librustc_mir/borrow_check/nll

The MIR-based region analysis consists of two major functions:

- `replace_regions_in_mir`, invoked first, has two jobs:
  - First, it analyzes the signature of the MIR and finds the set of
    regions that appear in the MIR signature (e.g., `'a` in `fn
    foo<'a>(&'a u32) { ... }`. These are called the "universal" or
    "free" regions -- in particular, they are the regions that
    [appear free][fvb] in the function body.
  - Second, it replaces all the regions from the function body with
    fresh inference variables. This is because (presently) those
    regions are the results of lexical region inference and hence are
    not of much interest. The intention is that -- eventually -- they
    will be "erased regions" (i.e., no information at all), since we
    don't be doing lexical region inference at all.
- `compute_regions`, invoked second: this is given as argument the
  results of move analysis. It has the job of computing values for all
  the inference variabes that `replace_regions_in_mir` introduced.
  - To do that, it first runs the [MIR type checker](#mirtypeck). This
    is basically a normal type-checker but specialized to MIR, which
    is much simpler than full Rust of course. Running the MIR type
    checker will however create **outlives constraints** between
    region variables (e.g., that one variable must outlive another
    one) to reflect the subtyping relationships that arise.
  - It also adds **liveness constraints** that arise from where variables
    are used.
  - More details to come, though the [NLL RFC] also includes fairly thorough
    (and hopefully readable) coverage.
  
[fvb]: background.html#free-vs-bound
[NLL RFC]: http://rust-lang.github.io/rfcs/2094-nll.html

## Universal regions

*to be written* -- explain the `UniversalRegions` type

## Region variables and constraints

*to be written* -- describe the `RegionInferenceContext` and
the role of `liveness_constraints` vs other `constraints`, plus

## Closures

<a name=mirtypeck>

## The MIR type-check

## Representing the "values" of a region variable

The value of a region can be thought of as a **set**; we call the
domain of this set a `RegionElement`. In the code, the value for all
regions is maintained in
[the `rustc_mir::borrow_check::nll::region_infer` module][ri]. For
each region we maintain a set storing what elements are present in its
value (to make this efficient, we give each kind of element an index,
the `RegionElementIndex`, and use sparse bitsets).

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
- There is an element `!1` for each skolemized region `!1`. This
  corresponds (intuitively) to some unknown set of other elements --
  for details on skolemization, see the section
  [skolemization and universes](#skol).
  
## Causal tracking

*to be written* -- describe how we can extend the values of a variable
 with causal tracking etc

<a name=skol>

## Skolemization and universes

(This section describes ongoing work that hasn't landed yet.)

From time to time we have to reason about regions that we can't
concretely know. For example, consider this program:

```rust
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

### Subtyping and skolemization

When we type-check `main`, and in particular the call `bar(foo)`, we
are going to wind up with a subtyping relationship like this one:

    fn(&'static u32) <: for<'a> fn(&'a u32)
    ----------------    -------------------
    the type of `foo`   the type `bar` expects
    
We handle this sort of subtyping by taking the variables that are
bound in the supertype and **skolemizing** them: this means that we
replace them with
[universally quantified](background.html#quantified)
representatives, written like `!1`. We call these regions "skolemized
regions" -- they represent, basically, "some unknown region".

Once we've done that replacement, we have the following types:

    fn(&'static u32) <: fn(&'!1 u32)
    
The key idea here is that this unknown region `'!1` is not related to
any other regions. So if we can prove that the subtyping relationship
is true for `'!1`, then it ought to be true for any region, which is
what we wanted. (This number `!1` is called a "universe", for reasons
we'll get into later.)

So let's work through what happens next. To check if two functions are
subtypes, we check if their arguments have the desired relationship
(fn arguments are [contravariant](./background.html#variance), so
we swap the left and right here):

    &'!1 u32 <: &'static u32

According to the basic subtyping rules for a reference, this will be
true if `'!1: 'static`. That is -- if "some unknown region `!1`" lives
outlives `'static`. Now, this *might* be true -- after all, `'!1`
could be `'static` -- but we don't *know* that it's true. So this
should yield up an error (eventually).

### Universes and skolemized region elements

But where does that error come from?  The way it happens is like this.
When we are constructing the region inference context, we can tell
from the type inference context how many skolemized variables exist
(the `InferCtxt` has an internal counter). For each of those, we
create a corresponding universal region variable `!n` and a "region
element" `skol(n)`. This corresponds to "some unknown set of other
elements". The value of `!n` is `{skol(n)}`.

At the same time, we also give each existential variable a
**universe** (also taken from the `InferCtxt`). This universe
determines which skolemized elements may appear in its value: For
example, a variable in universe U3 may name `skol(1)`, `skol(2)`, and
`skol(3)`, but not `skol(4)`. Note that the universe of an inference
variable controls what region elements **can** appear in its value; it
does not say region elements **will** appear.

### Skolemization and outlives constraints

In the region inference engine, outlives constraints have the form:

    V1: V2 @ P
    
where `V1` and `V2` are region indices, and hence map to some region
variable (which may be universally or existentially quantified). This
variable will have a universe, so let's call those universes `U(V1)`
and `U(V2)` respectively. (Actually, the only one we are going to care
about is `U(V1)`.)

When we encounter this constraint, the ordinary procedure is to start
a DFS from `P`. We keep walking so long as the nodes we are walking
are present in `value(V2)` and we add those nodes to `value(V1)`. If
we reach a return point, we add in any `end(X)` elements. That part
remains unchanged.

But then *after that* we want to iterate over the skolemized `skol(u)`
elements in V2 (each of those must be visible to `U(V2)`, but we
should be able to just assume that is true, we don't have to check
it). We have to ensure that `value(V1)` outlives each of those
skolemized elements.

Now there are two ways that could happen. First, if `U(V1)` can see
the universe `u` (i.e., `u <= U(V1)`), then we can just add `skol(u1)`
to `value(V1)` and be done. But if not, then we have to approximate:
we may not know what set of elements `skol(u1)` represents, but we
should be able to compute some sort of **upper bound** for it --
something that it is smaller than. For now, we'll just use `'static`
for that (since it is bigger than everything) -- in the future, we can
sometimes be smarter here (and in fact we have code for doing this
already in other contexts). Moreover, since `'static` is in U0, we
know that all variables can see it -- so basically if we find a that
`value(V2)` contains `skol(u)` for some universe `u` that `V1` can't
see, then we force `V1` to `'static`.

### Extending the "universal regions" check

After all constraints have been propagated, the NLL region inference
has one final check, where it goes over the values that wound up being
computed for each universal region and checks that they did not get
'too large'. In our case, we will go through each skolemized region
and check that it contains *only* the `skol(u)` element it is known to
outlive. (Later, we might be able to know that there are relationships
between two skolemized regions and take those into account, as we do
for universal regions from the fn signature.)

Put another way, the "universal regions" check can be considered to be
checking constraints like:

    {skol(1)}: V1
    
where `{skol(1)}` is like a constant set, and V1 is the variable we
made to represent the `!1` region.

## Back to our example

OK, so far so good. Now let's walk through what would happen with our
first example:

    fn(&'static u32) <: fn(&'!1 u32) @ P  // this point P is not imp't here

The region inference engine will create a region element domain like this:

    { CFG; end('static); skol(1) }
      ---  ------------  ------- from the universe `!1`
      |    'static is always in scope
      all points in the CFG; not especially relevant here 

It will always create two universal variables, one representing
`'static` and one representing `'!1`. Let's call them Vs and V1. They
will have initial values like so:

    Vs = { CFG; end('static) } // it is in U0, so can't name anything else
    V1 = { skol(1) }
    
From the subtyping constraint above, we would have an outlives constraint like

    '!1: 'static @ P

To process this, we would grow the value of V1 to include all of Vs:

    Vs = { CFG; end('static) }
    V1 = { CFG; end('static), skol(1) }

At that point, constraint propagation is done, because all the
outlives relationships are satisfied. Then we would go to the "check
universal regions" portion of the code, which would test that no
universal region grew too large.

In this case, `V1` *did* grow too large -- it is not known to outlive
`end('static)`, nor any of the CFG -- so we would report an error.

## Another example

What about this subtyping relationship?

    for<'a> fn(&'a u32, &'a u32)
        <:
    for<'b, 'c> fn(&'b u32, &'c u32)
    
Here we would skolemize the supertype, as before, yielding:    

    for<'a> fn(&'a u32, &'a u32)
        <:
    fn(&'!1 u32, &'!2 u32)
    
then we instantiate the variable on the left-hand side with an existential
in universe U2, yielding:

    fn(&'?3 u32, &'?3 u32) 
        <: 
    fn(&'!1 u32, &'!2 u32)
    
Then we break this down further:

    &'!1 u32 <: &'?3 u32
    &'!2 u32 <: &'?3 u32
    
and even further, yield up our region constraints:

    '!1: '?3
    '!2: '?3
    
Note that, in this case, both `'!1` and `'!2` have to outlive the
variable `'?3`, but the variable `'?3` is not forced to outlive
anything else. Therefore, it simply starts and ends as the empty set
of elements, and hence the type-check succeeds here.

(This should surprise you a little. It surprised me when I first
realized it. We are saying that if we are a fn that **needs both of
its arguments to have the same region**, we can accept being called
with **arguments with two distinct regions**. That seems intuitively
unsound. But in fact, it's fine, as I
[tried to explain in this issue on the Rust issue tracker long ago][ohdeargoditsallbroken].
The reason is that even if we get called with arguments of two
distinct lifetimes, those two lifetimes have some intersection (the
call itself), and that intersection can be our value of `'a` that we
use as the common lifetime of our arguments. -nmatsakis)

[ohdeargoditsallbroken]: https://github.com/rust-lang/rust/issues/32330#issuecomment-202536977

## Final example 

Let's look at one last example. We'll extend the previous one to have
a return type:

    for<'a> fn(&'a u32, &'a u32) -> &'a u32
        <:
    for<'b, 'c> fn(&'b u32, &'c u32) -> &'b u32
    
Despite seeming very similar to the previous example, this case is
going to get an error. That's good: the problem is that we've gone
from a fn that promises to return one of its two arguments, to a fn
that is promising to return the first one. That is unsound. Let's see how it plays out.

First, we skolemize the supertype:

    for<'a> fn(&'a u32, &'a u32) -> &'a u32
        <:
    fn(&'!1 u32, &'!2 u32) -> &'!1 u32
    
Then we instantiate the subtype with existentials (in U2):

    fn(&'?3 u32, &'?3 u32) -> &'?3 u32
        <:
    fn(&'!1 u32, &'!2 u32) -> &'!1 u32
    
And now we create the subtyping relationships:

    &'!1 u32 <: &'?3 u32 // arg 1
    &'!2 u32 <: &'?3 u32 // arg 2
    &'?3 u32 <: &'!1 u32 // return type
    
And finally the outlives relationships. Here, let V1, V2, and V3 be the variables
we assign to `!1`, `!2`, and `?3` respectively:

    V1: V3
    V2: V3
    V3: V1
    
Those variables will have these initial values:

    V1 in U1 = {skol(1)}
    V2 in U2 = {skol(2)}
    V3 in U2 = {}
    
Now because of the `V3: V1` constraint, we have to add `skol(1)` into `V3` (and indeed
it is visible from `V3`), so we get:

    V3 in U2 = {skol(1)}
    
then we have this constraint `V2: V3`, so we wind up having to enlarge
`V2` to include `skol(1)` (which it can also see):

    V2 in U2 = {skol(1), skol(2)}
    
Now contraint propagation is done, but when we check the outlives
relationships, we find that `V2` includes this new element `skol(1)`,
so we report an error.

