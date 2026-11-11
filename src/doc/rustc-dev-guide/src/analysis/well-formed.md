# Well-formedness

## What is well-formedness?

"Well-formed" means "correctly built"[^wf-history].
Something is _well-formed_ when its structure follows rules.
When we use this term in the Rust compiler we are concerned with establishing some kind of _internal consistency_.

## Well-formedness in Rust

To check that something is well-formed is to perform a "Well-formedness check".

In the Rust compiler there are two different forms of well-formedness checking:

- **Type-Level Term**[^terms][^terms-abbreviated] well-formedness check.
    - Also called "Term well-formedness" or "Term well-formedness checking".
    - Not a distinct analysis stage, this gets performed throughout analysis.
- **Item**[^items] well-formedness check (item-wfck.)
    - "Item-wfck" will often wind up requiring Terms be well-formed, but skips some areas.
    - Inner "Terms" can (incorrectly) get normalized first.
    - More coherent as a stage in the compiler than "term well-formedness" (which is performed in many places.)

See: [What Well-Formedness Isn't](#what-well-formedness-isnt).

## Well-formedness of type-level terms

Term well-formedness checking begins with building a list of things that need to be true for a term to be well-formed.
We call these "Obligations"[^obligations].

Type-Level Terms are considered well-formed when their associated obligations are satisfied by the trait solver.

### Obligations for well-formedness

Specific obligations are things like `String: Clone`, `A: usize`, or `<T as Iterator>::Item: Debug`.

On this page we show the split between obligations and terms/items as:

```rust,ignore
<terms or items>
---
<obligations>
```

Here is an example of a well-formed type-level term:

```rust,ignore
Vec<String>
---
// Obligations to fulfill
Vec<T> where T: Sized
// Trait solver says `String: Sized` is true, so this is well-formed.
Vec<String> where String: Sized
```

When we compute the obligations for `Vec<String>`, we'll find that `Vec<T>` generates the obligation `T: Sized`.
We substitute `T` with `String` in `Vec<String>`, so we find the obligation `String: Sized` which the trait solver will determine to be satisfied.

The following **is not** well-formed:

```rust,ignore
Vec<str>
---
// Obligations to fulfill
Vec<T> where T: Sized
// Trait solver says `str: Sized` is not true, so this is not well-formed.
Vec<str> where str: Sized
```

The above computes the obligation `T: Sized`, like before, but we substitute `T` for `str` in the instance of `Vec<str>` finding the obligation `str: sized`.
This obligation will be determined by the trait solver to be _unsatisfied_.

#### Determining obligations

In the compiler, obligations of terms are found through the [`obligations`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/wf/fn.obligations.html) function in the [term well-formedness module](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/wf/index.html).

#### Other obligations

Obligations are more than just trait and const generic bounds, but we've only mentioned these specific obligations so far as they are what we care about when we do "well-formedness checking" of terms.
See: [`PredicateKind`](https://doc.rust-lang.org/beta/nightly-rustc/rustc_type_ir/predicate_kind/enum.PredicateKind.html) and [`ClauseKind`](https://doc.rust-lang.org/beta/nightly-rustc/rustc_type_ir/predicate_kind/enum.ClauseKind.html) for a full list of obligations.

### We don't need normalization (yet)

[Normalization](../normalization.md) is the process of resolving [type aliases](../normalization.md#aliases) into their underlying type.

A type alias is considered well-formed if its where clauses are satisfied.
The underlying type undergoes well-formedness checking at most definition and instantiation sites, but there are exceptions.

### Const generic arguments

Term well-formedness is responsible for getting "type checking" obligations of const generic terms[^tyck-const-generics].
Let's look at the following use of const generics:

```rust,ignore
fn use_const_generics<const U: usize>() { /* ... */ }
// call site
use_const_generics::<6>();
---
// call site wfck obligations
const 6: usize
```

The call site will provide us with the obligation `6: usize` during well-formedness checking.
This obligation will be passed off to the trait solver just like any trait-style obligation, as the trait solver has more responsibilities than its name suggests.

## Well-formedness of items

Items are, generally speaking, "Things that get defined".
Item-wfck happens at the signature level for types and functions, methods, and definitions/implementations of traits.

```rust,ignore
// The `Vec<str>` is checked during item wfck
fn foo(_: Vec<str>) {
    // The `Vec<[u8]>` is not handled by item wfck as it's not in the signature
    let _: Vec<[u8]>
}
---
Vec<str>: Sized // Generated
Vec<[u8]>: Sized // Not done at item-wfck. Done elsewhere.
```

Item-wfck has more responsibilities than only collecting the obligations of its internal type-level terms and passing them to the trait solver.
We do not talk about all of these here, but they can be found at the individual `check_*` functions in [**the item-wfck module**](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/wfcheck/index.html).

<!-- FIXME: Expand more on item well-formedness that isn't const generic / trait bound obligation based. These are not special cases, but important points! -->

### Global and trivial bounds

<!-- TODO later: Cut this into its own page -->

Trait bounds are a common Obligation.
Global and Trivial trait bounds are kinds of trait bounds where we already have enough information to determine if they are true or false.
Item-wfck is responsible for finding and checking these bounds.

- **Global bounds** are, in the old solver, post-normalization bounds that don't contain any generic parameters (like `<T>` or `'a`) or bound variables (like `for<'b>`).
- **Trivial bounds** are bounds that do not need further normalization to determine if they're well-formed or not. <!-- TODO: check with lcnr if this is genuinely what a trivial bound is. -->

Consider the following function definition:

```rust,ignore
fn apartment_complex<T>(block: T, name: String) where String: Clone { /* ... */ }
---
String: Clone // Trivial & Global bound! There's no aliases to resolve.
// There could be bligations on T but we don't care about them here.
```

This produces a trait bound obligation `String: Clone` that is _Global_ (no generic parameters) and _Trivial_ (didn't require normalization to be well-formedness checked).
The trait solver doesn't need to be given any additional information for it to be able to make a judgment on the well-formedness of `String: Clone`.

False trivial bounds are simply trivial bounds that do not hold.
The following is a basic example:

```rust,ignore
fn apartment_simple<T>(block: T, name: String) where String: Copy { /* ... */ }
---
String: Copy // Trivial bound again, but this one is false!
```

Here we have a trivial bound that does not hold, because `String` is not `Copy`.

#### Trivial bounds are not always global

Trivial Bounds are not a subset of Global Bounds.
A trivial bound that isn't Global is `for<'a> String: Clone` (trivially true, has a bound variable) or `&'a str: Copy` (trivially false, has a generic parameter).

#### Item-wfck and trivial/global bounds

<!-- When cutting out the subsection on global/trivial bounds, keep this part on the well-formedness page. -->

When checking items are well-formed we will check that there are no trivially false global bounds.

## When we don't fully do well-formedness checking

Well-formedness checking is not a coherent "stage" of type checking.
There are many areas where well-formedness checking is performed, and some areas where we skip over well-formedness checking due to limitations in what kinds of analysis we can currently perform.
Ideally, we would never skip or defer well-formedness checking.

### We (sometimes) need normalization

There are places where normalization of an Item happens before its Terms have gone through well-formedness checking.
This is considered problematic as doing so allows some terms to [bypass term well-formedness checking entirely](https://github.com/rust-lang/rust/issues/100041).

### Trait objects

We do not require the where clauses of trait objects to be well-formed when determining if that trait object is well-formed.
These where clauses are proven when coercing into a trait object, but this remains a hole in well-formedness checking.

As an example, the following will compile because we don't have a point where we're constructing the trait object from a concrete type:

```rust,ignore
trait Trait
where
    for<'a> [u8]: Sized {}

fn foo(_: &dyn Trait) {}
---
// This doesn't end up being generated, because it happens within a trait object.
[u8]: Sized
```

The above should not compile because `[u8]: Sized`, but this won't be checked until actual use:

```rust
trait Trait
where
    for<'a> [u8]: Sized {}

fn foo(_: &dyn Trait) {}

// We still need to specify the bound here, otherwise `[u8]: Sized` _is_
// checked as an obligation.
impl Trait for u8 where for<'a> [u8]: Sized {}

fn main() {
    // No matter what we do, this boundary between concrete type and trait
    // object will produce the obligation `[u8]: Sized`, which will fail when
    // handed over to the trait solver.
    let object: Box<dyn Trait> = Box::new(42u8);
    foo(&object);
}
```

This exception does not apply to Const Generic Arguments in trait objects:

```rust,ignore
trait Trait<const N: usize> {}
fn foo<const B: bool>(_: &dyn Trait<B>) {}
---
const N: usize
const B: bool
N = B // Substitution
const B: usize + bool
```

The above doesn't compile, unlike the previous example we gave.
We're doing _some_ well-formedness checking here when it comes to the const generic arguments.

### Binders / higher-ranked types

Binders / Higher-Ranked Types reduce the amount well-formedness checking we do on a term, leaving well-formedness checking to when the bound is instantiated:

```rust,ignore
let _: for<'a> fn(Vec<[&'a ()]>);
---
// This doesn't end up being generated, because it happens within a HRB
[&'a ()]: Sized // slices aren't sized, this would fail!
```

Specifically, obligations involving variables from binders (`for<'a>`) are only checked when the binder is instantiated.
Some things are stilled checked under the `for<'a>`, but we still skip a lot of things.

A lot of unsoundness surrounds this behavior.
See: [#25860](https://github.com/rust-lang/rust/issues/25860), [#84591](https://github.com/rust-lang/rust/issues/84591).

Let's consider the following:

```rust,ignore
for<'a, 'b> fn(&'a &'b ())
```

The above HRB implies `'b: 'a` (a lifetime bound), rather than two completely separate lifetimes.
This is normal lifetime behavior, but during well-formedness checking we cannot prove that this bound is generally true[^horrible], so we skip it.

### Free type aliases

The right-hand side of Free Type Aliases[^fta] is not fully checked to be well-formed at the definition site, only the types of const generic arguments in the RHS are checked.

The following free type alias passes type checking, at time of writing:

```rust,ignore
type WorksButShouldNot = Vec<str>;
---
// This should fail! But we skip the RHS of free type aliases
str: Sized // Not generated
```

This shouldn't work, as both `T: Sized`, `str: Sized` are implied by `Vec<T>`.
This "passes" item-wfck because the RHS of a free type alias doesn't go through well-formedness checking _until it's used_.
Item-wfck is **deferred until use** for this specific case.

For Const Generics we still do a small amount of well-formedness checking at the definition site of a free type alias.
This is consistent with our current special-casing of const generic well-formedness checking when we skip over things like where bounds.

This means that the following, despite being of a similar form to the above example, fails as it should:

```rust,ignore
pub struct Consty<const A: bool>;
type Alias = Consty<42>;
---
// This *is* generated as an obligation, so this (correctly) fails.
42: bool // This is generated!
```

<!-- TODO: Link to something explaining the underlying "why" of the difference between const and trait well-formedness checking in FTAs, or eliminate that difference. Whatever comes first. -->

## "well-formed" or "wellformed"?

Prefer "well-formed" over "wellformed", as this is consistent with logic literature.
This also gets abbreviated to WF in other parts of the dev guide / docs.

## Informal usage

In conversation, contributors may refer to something as "well-formed" and not necessarily mean what we cover here because "well-formedness" is a general phrase associated with the correctness of formal structures.
This isn't necessarily in error, but it should be looked out for.

## What well-formedness isn't

Well-formedness checking is not "number of parameters" or "parameter type" checking[^kind-checking].
Neither term well-formedness checking nor item-wfck is concerned with if a type with 2 parameters has 1 or 3 types applied to it (assuming no defaults), or if a const generic parameter has a type applied to it.
These kinds of problems will get handled during HIR-ty Lowering[^hir-ty-lower], not wfck.

Well-formedness doesn't check or validate lifetimes, this is handled in [MIR](../borrow-check.md).

Well-formedness in the Rust compiler doesn't correspond to "correct syntax" as it does in logic.
The term has a history of general use in a mathematical context of "follows a given set of rules".
In Rust, our original usage was closer to "this thing is internally consistent" with respect to the bounds on a type in places such as the original [clarification on projections and well-formedness RFC](https://github.com/rust-lang/rfcs/blob/master/text/1214-projections-lifetimes-and-wf.md).

[^obligations]: These get referred to as Obligations, Requirements, or Constraints in the documentation. Preferred term is "obligations", as this matches the suffix of the type and the names of relevant functions. In future, this may be superseded by the new solver's term "Goal".
[^wf-history]: In linguistics this is "grammatically correct", in logic it is "syntactically correct", and in casual mathematician use it can be read as a more general "follows the rules we set for this domain".
[^horrible]: Instead, this bound is checked during "MIR borrowck" when the lifetimes are instantiated.
[^fta]: Type aliases not associated with anything, i.e. a module-level `type Alias = Vec<u8>;`.
[^items]: "Definition" style things in rust, See the [glossary](../appendix/glossary.md).
[^terms]: AKA Type expressions and subexpressions in the general sense, not a specific struct or enum in the rust compiler. See the [glossary](../appendix/glossary.md).
[^terms-abbreviated]: Abbreviated as "Terms" on this page in some areas.
[^kind-checking]: AKA "kind checking", as we might see in languages like Haskell.
[^hir-ty-lower]: <https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/hir_ty_lowering/index.html>
[^tyck-const-generics]: #checking-types-of-const-arguments
