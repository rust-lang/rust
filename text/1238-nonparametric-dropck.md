- Feature Name: dropck_parametricity
- Start Date: 2015-08-05
- RFC PR: https://github.com/rust-lang/rfcs/pull/1238/
- Rust Issue: https://github.com/rust-lang/rust/issues/28498

# Summary

Revise the Drop Check (`dropck`) part of Rust's static analyses in two
ways.  In the context of this RFC, these revisions are respectively
named `cannot-assume-parametricity` and `unguarded-escape-hatch`.

  1. `cannot-assume-parametricity` (CAP): Make `dropck` analysis stop
     relying on parametricity of type-parameters.

  2. `unguarded-escape-hatch` (UGEH): Add an attribute (with some name
     starting with "unsafe") that a library designer can attach to a
     `drop` implementation that will allow a destructor to side-step
     the `dropck`'s constraints (unsafely).

# Motivation

## Background: Parametricity in `dropck`

The Drop Check rule (`dropck`) for [Sound Generic Drop][] relies on a
reasoning process that needs to infer that the behavior of a
polymorphic function (e.g. `fn foo<T>`) does not depend on the
concrete type instantiations of any of its *unbounded* type parameters
(e.g. `T` in `fn foo<T>`), at least beyond the behavior of the
destructor (if any) for those type parameters.

[Sound Generic Drop]: https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md

This property is a (weakened) form of a property known in academic
circles as *Parametricity*.
(See e.g. [Reynolds, IFIP 1983][Rey83], [Wadler, FPCA 1989][Wad89].)

 * Parametricity, in this context, essentially says that the compiler
   can reason about the body of `foo` (and the subroutines that `foo`
   invokes) without having to think about the particular concrete
   types that the type parameter `T` is instantiated with.
   `foo` cannot do anything with a `t: T` except:

   1. move `t` to some other owner expecting a `T` or,

   2. drop `t`, running its destructor and freeing associated resources.

 * For example, this allows the compiler to deduce that even if `T` is
   instantiated with a concrete type like `&Vec<u32>`, the body of
   `foo` cannot actually read any `u32` data out of the vector. More
   details about this are available on the [Sound Generic Drop][] RFC.

## "Mistakes were made"

The parametricity-based reasoning in the
[Drop Check analysis][Sound Generic Drop] (`dropck`) was clever, but
fragile and unproven.

 * Regarding its fragility, it has been shown to have
   [bugs][parametricity-insufficient]; in particular, parametricity is
   a necessary but *not* sufficient condition to justify the
   inferences that `dropck` makes.

 * Regarding its unproven nature, `dropck` violated the heuristic in
   Rust's design to not incorporate ideas unless those ideas had
   already been proven effective elsewhere.

[parametricity-insufficient]: https://github.com/rust-lang/rust/issues/26656

These issues might alone provide motivation for ratcheting back on
`dropck`'s rules in the short term, putting in a more conservative
rule in the stable release channel while allowing experimentation with
more-aggressive feature-gated rules in the development nightly release
channel.
   
However, there is also a specific reason why we want to ratchet back
on the `dropck` analysis as soon as possible.

## Impl specialization is inherently non-parametric

The parametricity requirement in the Drop Check rule over-restricts
the design space for future language changes.

In particular, the [impl specialization] RFC describes a language
change that will allow the invocation of a polymorphic function `f` to
end up in different sequences of code based solely on the concrete
type of `T`, *even* when `T` has no trait bounds within its
declaration in `f`.

[impl specialization]: https://github.com/rust-lang/rfcs/pull/1210

# Detailed design

Revise the Drop Check (`dropck`) part of Rust's static analyses in two
ways.  In the context of this RFC, these revisions are respectively
named `cannot-assume-parametricity` (CAP) and `unguarded-escape-hatch` (UGEH).

Though the revisions are given distinct names, they both fall under
the feature gate `dropck_parametricity`. (Note however that this
might be irrelevant to CAP; see [CAP stabilization details][]).

## cannot-assume-parametricity

The heart of CAP is this: make `dropck` analysis stop relying on
parametricity of type-parameters.

### Changes to the Drop-Check Rule

The Drop-Check Rule (both in its original form and as revised here)
dicates when a lifetime `'a` must strictly outlive some value `v`,
where `v` owns data of type `D`; the rule gave two circumstances where
`'a` must strictly outlive the scope of `v`.

 * The first circumstance (`D` is directly instantiated at `'a`)
   remains unchanged by this RFC.

 * The second circumstance (`D` has some type parameter with
   trait-provided methods, i.e. that could be invoked within `Drop`)
   is broadened by this RFC to simply say "`D` has some type
   parameter."

That is, under the changes of this RFC, whether the type parameter has
a trait-bound is irrelevant to the Drop-Check Rule. The reason is that
any type parameter, regardless of whether it has a trait bound or not,
may end up participating in [impl specialization], and thus could
expose an otherwise invisible reference `&'a AlreadyDroppedData`.

`cannot-assume-parametricity` is a breaking change, since the language
will start assuming that a destructor for a data-type definition such
as `struct Parametri<C>` may read from data held in its `C` parameter,
even though the `fn drop` formerly appeared to be parametric with
respect to `C`. This will cause `rustc` to reject code that it had
previously accepted (below are some examples that
[continue to work][examples-continue-to-work] and
some that [start being rejected][examples-start-reject]).

### CAP stabilization details
[CAP stabilization details]: #cap-stabilization-details

`cannot-assume-parametricity` will be incorporated into the beta
and stable Rust channels, to ensure that destructor code atop
stable channels in the wild stop relying on parametricity as soon
as possible. This will enable new language features such as
[impl specialization].

 * It is not yet clear whether it is feasible to include a warning
   cycle for CAP.

 * For now, this RFC is proposing to remove the parts of Drop-Check
   that attempted to prove that the `impl<T> Drop` was parametric with
   respect to `T`. This would mean that there would be more warning
   cycle; `dropck` would simply start rejecting more code.
   There would be no way to opt back into the old `dropck` rules.

 * (However, during implementation of this change, we should
    double-check whether a warning-cycle is in fact feasible.)

## unguarded-escape-hatch

The heart of `unguarded-escape-hatch` (UGEH) is this: Provide a new,
unsafe (and unstable) attribute-based escape hatch for use in the
standard library for cases where Drop Check is too strict.

### Why we need an escape hatch

The original motivation for the parametricity special-case in the
original Drop-Check rule was due to an observation that collection
types such as `TypedArena<T>` or `Vec<T>` were often used to
contain values that wanted to refer to each other.

An example would be an element type like
`struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);`, and then
instantiations of `TypedArena<Concrete>` or `Vec<Concrete>`.
This pattern has been used within `rustc`, for example,
to store elements of a linked structure within an arena.

Without the parametricity special-case, the existence of a destructor
on `TypedArena<T>` or `Vec<T>` led the Drop-Check analysis to conclude
that those destructors might hypothetically read from the references
held within `T` -- forcing `dropck` to reject those destructors.

(Note that `Concrete` itself has no destructor; if it did, then
`dropck`, both as originally stated and under the changes of this RFC,
*would* force the `'a` parameter of any instance to strictly outlive
the instance value, thus ruling out cross-references in the same
`TypedArena` or `Vec`.)

Of course, the whole point of this RFC is that using parametricity as
the escape hatch seems like it does not suffice. But we still need
*some* escape hatch.

### The new escape hatch: an unsafe attribute

This leads us to the second component of the RFC, `unguarded-escape-hatch` (UGEH):
Add an attribute (with a name starting with "unsafe") that a library
designer can attach to a `drop` implementation that will allow a
destructor to side-step the `dropck`'s constraints (unsafely).

This RFC proposes the attribute name `unsafe_destructor_blind_to_params`.
This name was specifically chosen to be long and ugly; see
[UGEH stabilization details] for further discussion.

Much like the `unsafe_destructor` attribute that we had in the past,
this attribute relies on the programmer to ensure that the destructor
cannot actually be used unsoundly. It states an (unproven) assumption
that the given implementation of `drop` (and all functions that this
 `drop` may transitively call) will never read or modify a value of
any type parameter, apart from the trivial operations of either
dropping the value or moving the value from one location to another.

 * (In particular, it certainly must not dereference any `&`-reference
   within such a value, though this RFC is adopts a somewhat stronger
   requirement to encourage the attribute to only be used for the
   limited case of parametric collection types, where one need not do
   anything more than move or drop values.)

The above assumption must hold regardless of what impact
[impl specialization][] has on the resolution of all function calls.

### UGEH stabilization details
[UGEH stabilization details]: #ugeh-stabilization-details

The proposed attribute is only a *short-term* patch to work-around a
bug exposed by the combination of two desirable features (namely
[impl specialization] and [`dropck`][Sound Generic Drop]).

In particular, using the attribute in cases where control-flow in the
destructor can reach functions that may be specialized on a
type-parameter `T` may expose the system to use-after-free scenarios
or other unsound conditions. This may a non-trivial thing for the
programmer to prove.

 * Short term strategy: The working assumption of this RFC is that the
   standard library developers will use the proposed attribute in
   cases where the destructor *is* parametric with respect to all type
   parameters, even though the compiler cannot currently prove this to
   be the case.

   The new attribute will be restricted to non-stable channels, like
   any other new feature under a feature-gate.

 * Long term strategy: This RFC does not make any formal guarantees
   about the long-term strategy for including an escape hatch. In
   particular, this RFC does *not* propose that we stabilize the
   proposed attribute

   It may be possible for future language changes to allow us to
   directly express the necessary parametricity properties.
   See further discussion in the [continue supporting parametricity][] alternative.

   The suggested attribute name (`unsafe_destructor_blind_to_params`
   above) was deliberately selected to be long and ugly, in order to
   discourage it from being stabilized in the future without at least
   some significant discussion. (Likewise, the acronym "UGEH" was
   chosen for its likely pronounciation "ugh", again a reminder that
   we do not *want* to adopt this approach for the long term.)


## Examples of code changes under the RFC

This section shows some code examples, starting with code that works
today and must continue to work tomorrow, then showing an example of
code that will start being rejected, and ending with an example of the
UGEH attribute.

### Examples of code that must continue to work
[examples-continue-to-work]: #examples-of-code-that-must-continue-to-work

Here is some code that works today and must continue to work in the future:

```rust
use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

fn main() {
    let mut data = Vec::new();
    data.push(Concrete(0, Cell::new(None)));
    data.push(Concrete(0, Cell::new(None)));

    data[0].1.set(Some(&data[1]));
    data[1].1.set(Some(&data[0]));
}
```

In the above, we are building up a vector, pushing `Concrete` elements
onto it, and then later linking those concrete elements together via
optional references held in a cell in each concrete element.

We can even wrap the vector in a struct that holds it.  This also must
continue to work (and will do so under this RFC); such structural
composition is a common idiom in Rust code.

```rust
use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
```

### Examples of code that will start to be rejected
[examples-start-reject]: #examples-of-code-that-will-start-to-be-rejected

The main change injected by this RFC is this: due to `cannot-assume-parametricity`,
an attempt to add a destructor to the `struct Foo` above will cause the
code above to be rejected, because we will assume that the destructor for `Foo`
may invoke methods on the concrete elements that dereferences their links.

Thus, this code will be rejected:

```rust
use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

// This is the new `impl Drop`
impl<T> Drop for Foo<T> {
    fn drop(&mut self) { }
}

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
```

NOTE: Based on a preliminary crater run, it seems that mixing together
destructors with this sort of cyclic structure is sufficiently rare
that *no* crates on `crates.io` actually regressed under the new rule:
everything that compiled before the change continued to compile after
it.

### Example of the unguarded-escape-hatch
[examples-escape-hatch]: #example-of-the-unguarded-escape-hatch

If the developer of `Foo` has access to the feature-gated
escape-hatch, and is willing to assert that the destructor for `Foo`
does nothing with the links in the data, then the developer can work
around the above rejection of the code by adding the corresponding
attribute.

```rust
#![feature(dropck_parametricity)]
use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

impl<T> Drop for Foo<T> {
    #[unsafe_destructor_blind_to_params] // This is the UGEH attribute
    fn drop(&mut self) { }
}

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
```

# Drawbacks

As should be clear by the tone of this RFC, the
`unguarded-escape-hatch` is clearly a hack. It is subtle and unsafe,
just as `unsafe_destructor` was (and for the most part, the whole
point of [Sound Generic Drop][] was to remove `unsafe_destructor` from
the language).

 * However, the expectation is that most clients will have no need to
   ever use the `unguarded-escape-hatch`.

 * It may suffice to use the escape hatch solely within the collection
   types of `libstd`.

 * Otherwise, if clients outside of `libstd` determine that they *do*
   need to be able to write destructors that need to bypass `dropck`
   safely, then we can (and *should*) investigate one of the
   [sound alternatives][continue supporting parametricity], rather
   than stabilize the unsafe hackish escape hatch..

# Alternatives
[alternatives]: #alternatives

## CAP without UGEH

One might consider adopting `cannot-assume-parametricity` without
`unguarded-escape-hatch`.  However, unless some other sort of escape
hatch were added, this path would break much more code.

## UGEH for lifetime parameters

Since we're already being unsafe here, one might consider having
the `unsafe_destructor_blind_to_params` apply to lifetime parameters
as well as type parameters.

However, given that the `unsafe_destructor_blind_to_params` attribute
is only intended as a short-term band-aid (see
[UGEH stabilization details][]) it seems better to just make it only as
broad as it needs to be (and no broader).

## "Sort-of Guarded" Escape Hatch

We could add the escape hatch but continue employing the current
dropck analysis to it. This would essentially mean that code would have
to apply the unsafe attribute to be considered for parametricity, but
if there were obvious problems (namely, if the type parameter had a trait bound)
then the attempt to opt into parametricity would be ignored and the
strict ordering restrictions on the lifetimes would be imposed.

I only mention this because it occurred to me in passing; I do not
really think it has much of a benefit. It would potentially lead
someone to think that their code has been proven sound (since the
`dropck` would catch some mistakes in programmer reasoning) but the
pitfalls with respect to specialization would remain.

## Continue Supporting Parametricity
[continue supporting parametricity]: #continue-supporting-parametricity
There may be ways to revise the language so that functions can declare
that they must be parametric with respect to their type parameters.
Here we sketch two potential ideas for how one might do this, mostly to
give a hint of why this is not a trivial change to the language.

Neither design is likely to be adopted, at least as described here,
because both of them impose significant burdens on implementors of
parametric destructors, as we will see.

(Also, if we go down this path, we will need to fix other bugs in the
Drop Check rule, where, as previously noted, parametricity is a
[necessary but *insufficient* condition][parametricity-insufficient] for soundness.)

### Parametricity via effect-system attributes

One feature of the [impl specialization] RFC is that all functions that
can be specialized must be declared as such, via the `default` keyword.

This leads us to one way that a function could declare that its body
must not be allows to call into specialized methods: an attribute like
`#[unspecialized]`. The `#[unspecialized]` attribute, when applied to
a function `fn foo()`, would mean two things:

 * `foo` is not allowed to call any functions that have the `default` keyword.

 * `foo` is only allowed to call functions that are also marked `#[unspecialized]`

All `fn drop` methods would be required to be `#[unspecialized]`.

It is the second bullet that makes this an ad-hoc effect system: it provides
a recursive property ensuring that during the extent of the call to `foo`,
we will never invoke a function marked as `default` (and therefore, I *think*,
will never even potentially invoke a method that has been specialized).

It is also this second bullet that represents a signficant burden on
the destructor implementor. In particular, it immediately rules out
using any library routine unless that routine has been marked as
`#[unspecialized]`. The attribute is unlikely to be included on any
function unless the its developer is making a destructor that calls it
in tandem.

### Parametricity via some `?`-bound

Another approach starts from another angle: As described earlier,
parametricity in `dropck` is the requirement that `fn drop` cannot do
anything with a `t: T` (where `T` is some relevant type parameter)
except:

   1. move `t` to some other owner expecting a `T` or,

   2. drop `t`, running its destructor and freeing associated resources.

So, perhaps it would be more natural to express this requirement
via a bound.

We would start with the assumption that functions may be
non-parametric (and thus their implementations may be specialized to
specific types).

But then if you want to declare a function as having a stronger
constraint on its behavior (and thus expanding its potential callers
to ones that require parametricity), you could add a bound `T: ?Special`.

The Drop-check rule would treat `T: ?Special` type-parameters as parametric,
and other type-parameters as non-parametric.

The marker trait `Special` would be an OIBIT that all sized types would get.

Any expression in the context of a type-parameter binding of the form
`<T: ?Special>` would not be allowed to call any `default` method
where `T` could affect the specialization process.

(The careful reader will probably notice the potential sleight-of-hand
here: is this really any different from the effect-system attributes
proposed earlier? Perhaps not, though it seems likely that the finer
grain parameter-specific treatment proposed here is more expressive,
at least in theory.)

Like the previous proposal, this design represents a significant
burden on the destructor implementor: Again, the `T: ?Special`
attribute is unlikely to be included on any function unless the its
developer is making a destructor that calls it in tandem.

# Unresolved questions

 * What name to use for the attribute?
   Is `unsafe_destructor_blind_to_params` sufficiently long and ugly? ;)

 * What is the real long-term plan?

 * Should we consider merging the discussion of alternatives
   into the [impl specialization] RFC?

# Bibliography

### Reynolds
[Rey83]: #reynolds
John C. Reynolds. "Types, abstraction and parametric polymorphism". IFIP 1983
http://www.cse.chalmers.se/edu/year/2010/course/DAT140_Types/Reynolds_typesabpara.pdf

### Wadler
[Wad89]: #wadler
Philip Wadler. "Theorems for free!". FPCA 1989
http://ttic.uchicago.edu/~dreyer/course/papers/wadler.pdf

