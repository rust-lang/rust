# Opaque types region inference restrictions

In this chapter we discuss the various restrictions we impose on the generic arguments of
opaque types when defining their hidden types
`Opaque<'a, 'b, .., A, B, ..> := SomeHiddenType`.

These restrictions are implemented in borrow checking ([Source][source-borrowck-opaque])
as it is the final step opaque types inference.

[source-borrowck-opaque]: https://github.com/rust-lang/rust/blob/435b5255148617128f0a9b17bacd3cc10e032b23/compiler/rustc_borrowck/src/region_infer/opaque_types.rs

## Background: type and const generic arguments
For type arguments, two restrictions are necessary: each type argument must be
(1) a type parameter and
(2) is unique among the generic arguments.
The same is applied to const arguments.

Example of case (1):
```rust
type Opaque<X> = impl Sized;

// `T` is a type parameter.
// Opaque<T> := ();
fn good<T>() -> Opaque<T> {}

// `()` is not a type parameter.
// Opaque<()> := ();
fn bad() -> Opaque<()> {} //~ ERROR
```

Example of case (2):
```rust
type Opaque<X, Y> = impl Sized;

// `T` and `U` are unique in the generic args.
// Opaque<T, U> := T;
fn good<T, U>(t: T, _u: U) -> Opaque<T, U> { t }

// `T` appears twice in the generic args.
// Opaque<T, T> := T;
fn bad<T>(t: T) -> Opaque<T, T> { t } //~ ERROR
```
**Motivation:** In the first case `Opaque<()> := ()`, the hidden type is ambiguous because
it is compatible with two different interpretaions: `Opaque<X> := X` and `Opaque<X> := ()`.
Similarly for the second case `Opaque<T, T> := T`, it is ambiguous whether it should be
interpreted as `Opaque<X, Y> := X` or as `Opaque<X, Y> := Y`.
Because of this ambiguity, both cases are rejected as invalid defining uses.

## Uniqueness restriction

Each lifetime argument must be unique in the arguments list and must not be `'static`.
This is in order to avoid an ambiguity with hidden type inference similar to the case of
type parameters.
For example, the invalid defining use below `Opaque<'static> := Inv<'static>` is compatible with
both `Opaque<'x> := Inv<'static>` and `Opaque<'x> := Inv<'x>`.

```rust
type Opaque<'x> = impl Sized + 'x;
type Inv<'a> = Option<*mut &'a ()>;

fn good<'a>() -> Opaque<'a> { Inv::<'static>::None }

fn bad() -> Opaque<'static> { Inv::<'static>::None }
//~^ ERROR
```

```rust
type Opaque<'x, 'y> = impl Trait<'x, 'y>;

fn good<'a, 'b>() -> Opaque<'a, 'b> {}

fn bad<'a>() -> Opaque<'a, 'a> {}
//~^ ERROR
```

**Semantic lifetime equality:**
One complexity with lifetimes compared to type parameters is that
two lifetimes that are syntactically different may be semantically equal.
Therefore, we need to be cautious when verifying that the lifetimes are unique.

```rust
// This is also invalid because `'a` is *semantically* equal to `'static`.
fn still_bad_1<'a: 'static>() -> Opaque<'a> {}
//~^ Should error!

// This is also invalid because `'a` and `'b` are *semantically* equal.
fn still_bad_2<'a: 'b, 'b: 'a>() -> Opaque<'a, 'b> {}
//~^ Should error!
```

## An exception to uniqueness rule

An exception to the uniqueness rule above is when the bounds at the opaque type's definition require
a lifetime parameter to be equal to another one or to the `'static` lifetime.
```rust
// The definition requires `'x` to be equal to `'static`.
type Opaque<'x: 'static> = impl Sized + 'x;

fn good() -> Opaque<'static> {}
```

**Motivation:** an attempt to implement the uniqueness restriction for RPITs resulted in a
[breakage found by crater]( https://github.com/rust-lang/rust/pull/112842#issuecomment-1610057887).
This can be mitigated by this exception to the rule.
An example of the code that would otherwise break:
```rust
struct Type<'a>(&'a ());
impl<'a> Type<'a> {
    // `'b == 'a`
    fn do_stuff<'b: 'a>(&'b self) -> impl Trait<'a, 'b> {}
}
```

**Why this is correct:** for such a defining use like `Opaque<'a, 'a> := &'a str`,
it can be interpreted in either way—either as `Opaque<'x, 'y> := &'x str` or as
`Opaque<'x, 'y> := &'y str` and it wouldn't matter because every use of `Opaque`
will guarantee that both parameters are equal as per the well-formedness rules.

## Universal lifetimes restriction

Only universally quantified lifetimes are allowed in the opaque type arguments.
This includes lifetime parameters and placeholders.

```rust
type Opaque<'x> = impl Sized + 'x;

fn test<'a>() -> Opaque<'a> {
    // `Opaque<'empty> := ()`
    let _: Opaque<'_> = ();
    //~^ ERROR
}
```

**Motivation:**
This makes the lifetime and type arguments behave consistently but this is only as a bonus.
The real reason behind this restriction is purely technical, as the [member constraints] algorithm
faces a fundamental limitation:
When encountering an opaque type definition `Opaque<'?1> := &'?2 u8`,
a member constraint `'?2 member-of ['static, '?1]` is registered.
In order for the algorithm to pick the right choice, the *complete* set of "outlives" relationships
between the choice regions `['static, '?1]` must already be known *before* doing the region
inference. This can be satisfied only if each choice region is either:
1. a universal region, i.e. `RegionKind::Re{EarlyParam,LateParam,Placeholder,Static}`,
because the relations between universal regions are completely known, prior to region inference,
from the explicit and implied bounds.
1. or an existential region that is "strictly equal" to a universal region.
Strict lifetime equality is defined below and is required here because it is the only type of
equality that can be evaluated prior to full region inference.

**Strict lifetime equality:**
We say that two lifetimes are strictly equal if there are bidirectional outlives constraints
between them. In NLL terms, this means the lifetimes are part of the same [SCC].
Importantly this type of equality can be evaluated prior to full region inference
(but of course after constraint collection).
The other type of equality is when region inference ends up giving two lifetimes variables
the same value even if they are not strictly equal.
See [#113971] for how we used to conflate the difference.

[#113971]: https://github.com/rust-lang/rust/issues/113971
[SCC]: https://en.wikipedia.org/wiki/Strongly_connected_component
[member constraints]: ./region_inference/member_constraints.md

**interaction with "once modulo regions" restriction**
In the example above, note the opaque type in the signature is `Opaque<'a>` and the one in the
invalid defining use is `Opaque<'empty>`.
In the proposed MiniTAIT plan, namely the ["once modulo regions"][#116935] rule,
we already disallow this.
Although it might appear that "universal lifetimes" restriction becomes redundant as it logically
follows from "MiniTAIT" restrictions, the subsequent related discussion on lifetime equality and
closures remains relevant.

[#116935]: https://github.com/rust-lang/rust/pull/116935


## Closure restrictions

When the opaque type is defined in a closure/coroutine/inline-const body, universal lifetimes that
are "external" to the closure are not allowed in the opaque type arguments.
External regions are defined in [`RegionClassification::External`][source-external-region]

[source-external-region]: https://github.com/rust-lang/rust/blob/caf730043232affb6b10d1393895998cb4968520/compiler/rustc_borrowck/src/universal_regions.rs#L201.

Example: (This one happens to compile in the current nightly but more practical examples are
already rejected with confusing errors.)
```rust
type Opaque<'x> = impl Sized + 'x;

fn test<'a>() -> Opaque<'a> {
    let _ = || {
        // `'a` is external to the closure
        let _: Opaque<'a> = ();
        //~^ Should be an error!
    };
    ()
}
```

**Motivation:** 
In closure bodies, external lifetimes, although being categorized as "universal" lifetimes,
behave more like existential lifetimes in that the relations between them are not known ahead of
time, instead their values are inferred just like existential lifetimes and the requirements are
propagated back to the parent fn. This breaks the member constraints algorithm as described above:
> In order for the algorithm to pick the right choice, the complete set of “outlives” relationships
between the choice regions `['static, '?1]` must already be known before doing the region inference

Here is an example that details how :

```rust
type Opaque<'x, 'y> = impl Sized;

// 
fn test<'a, 'b>(s: &'a str) -> impl FnOnce() -> Opaque<'a, 'b> {
    move || { s }
    //~^ ERROR hidden type for `Opaque<'_, '_>` captures lifetime that does not appear in bounds
}

// The above closure body is desugared into something like:
fn test::{closure#0}(_upvar: &'?8 str) -> Opaque<'?6, '?7> {
    return _upvar
}

// where `['?8, '?6, ?7]` are universal lifetimes *external* to the closure.
// There are no known relations between them *inside* the closure.
// But in the parent fn it is known that `'?6: '?8`.
//
// When encountering an opaque definition `Opaque<'?6, '?7> := &'8 str`,
// The member constraints algorithm does not know enough to safely make `?8 = '?6`.
// For this reason, it errors with a sensible message:
// "hidden type captures lifetime that does not appear in bounds".
```

Without these restrictions, error messages are confusing and, more importantly, there is a risk that
we accept code that would likely break in the future because member constraints are super broken
in closures.

**Output types:**
I believe the most common scenario where this causes issues in real-world code is with
closure/async-block output types. It is worth noting that there is a discrepancy between closures
and async blocks that further demonstrates this issue and is attributed to the
[hack of `replace_opaque_types_with_inference_vars`][source-replace-opaques],
which is applied to futures only.

[source-replace-opaques]: https://github.com/rust-lang/rust/blob/9cf18e98f82d85fa41141391d54485b8747da46f/compiler/rustc_hir_typeck/src/closure.rs#L743

```rust
type Opaque<'x> = impl Sized + 'x;
fn test<'a>() -> impl FnOnce() -> Opaque<'a> {
    // Output type of the closure is Opaque<'a>
    // -> hidden type definition happens *inside* the closure
    // -> rejected.
    move || {}
    //~^ ERROR expected generic lifetime parameter, found `'_`
}
```
```rust
use std::future::Future;
type Opaque<'x> = impl Sized + 'x;
fn test<'a>() -> impl Future<Output = Opaque<'a>> {
    // Output type of the async block is unit `()`
    // -> hidden type definition happens in the parent fn
    // -> accepted.
    async move {}
}
```
