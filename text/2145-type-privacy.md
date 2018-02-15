- Feature Name: N/A
- Start Date: 2017-09-09
- RFC PR: [rust-lang/rfcs#2145](https://github.com/rust-lang/rfcs/pull/2145)
- Rust Issue: [rust-lang/rust#48054](https://github.com/rust-lang/rust/issues/48054)

# Summary
[summary]: #summary

Type privacy rules are documented.  
Private-in-public errors are relaxed and turned into lints.

# Motivation
[motivation]: #motivation

Type privacy is implemented, but its rules still need to be documentated and
explained.

Private-in-public checker is the previous incarnation of type privacy that
still exists in the compiler.  
Experience shows that private-in-public errors are often considered
non-intuitive, despite the rules being simple and sufficiently clear when
explained.  
People often expect private-in-public checker to check something it is not
supposed to check and otherwise, allow code that isn't supposed to be allowed.
This creates a source of confusion.

With type privacy implemented, private-in-public errors are no longer strictly
necessary, so they can be removed from the language, thus removing the source of
confusion.  
However diagnosing "private-in-public" situations early can still help
programmers to prevent most of client-side type privacy errors, so
"private-in-public" diagnostics can be turned into lints instead of being
completely removed.  
Lints, unlike errors, can use heuristics, so "private-in-public" diagnostics can
match programmer's intuition closer now by using reachability-based heuristics
instead of just local `pub` annotations.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Type privacy

Type privacy ensures that a type private to some module cannot be used outside
of this module (unless anonymized) without a privacy error.  
This is similar to more familiar name privacy ensuring that private items or
fields can't be *named* outside of their module without a privacy error.

"Using" a type means either explicitly naming it (maybe through `type` aliases),
or obtaining a value of that type.

```rust
mod m {
    struct Priv; // This is a type private to module `m`

    // OK, public alias to the private type
    pub type Alias = Priv;
    pub type AliasOpt = Option<Priv>;

    // OK, public function returnning a value of the private type
    pub fn get_value() -> Priv { ... }
}

// ERROR, can't name private type `m::Priv` outside of its module
type X = m::Alias;

// A type is considered private even if its primary component (type constructor)
// is public, but it has private generic arguments.
// ERROR, can't name private type `Option<m::Priv>` outside of its module
type X = m::AliasOpt;

fn main() {
    // ERROR, can't have a value of private type `m::Priv` outside of its module
    let x = m::get_value();
}
```

Type privacy ensures that a private type is an implementation detail of its
module and you can always change it in any way (e.g. add or remove methods,
add or remove trait implementations) without requiring any changes in other
modules.

Let's imagine for a minute that type privacy doesn't work and you can name
a private type `Priv` through an alias or obtain its values outside of its
module.  
Then let's assume that this type implements some trait `Trait` at the moment.
Now foreign code can freely define functions like
```rust
fn require_trait_value<T: Trait>(arg: T) { ... }
fn require_trait_type<T: Trait>() { ... }
```
and pass `Priv` to them
```rust
require_trait_value(value_of_priv);
require_trait_type::<AliasOfPriv>();
```
, so it becomes a *requirement* for `Priv` to implement `Trait` and we can't
remove it anymore.  
Type privacy helps to avoid such unintended requirements.

The sentence introducing type privacy contains a clarification - "unless
anonymized".  
It means that private types can be leaked into other modules through trait
objects (dynamically anonymized), or `impl Trait` (statically anonymized),
or usual generics (statically anonymized as well).
```rust
struct Priv;

// By defining funcions like these you explicitly give a promise that they will
// always return something implementing `Trait`, maybe `Priv`, maybe some other
// type (this is an implementation detail).
impl Trait for Priv {}
pub fn leak_anonymized1() -> Box<Trait> { Box::new(Priv) }
pub fn leak_anonymized2() -> impl Trait { Priv }

// Here some code outside of our module (in `liballoc`) works with objects of
// our private type, but knows only that they are `Clone`, the specific
// container element's type is anonymized for code in `liballoc`.
impl Clone for Priv {}
let my_vec: Vec<Priv> = vec![Priv, Priv, Priv];
let my_vec2 = my_vec.clone();
```

The rules for type privacy work for traits as well, e.g. you won't be able to
do this when trait aliases are implemented
```rust
mod m {
    trait PrivTr {}
    pub trait Alias = PrivTr;
}

// ERROR, can't name private trait `m::PrivTr` outside of its module
fn f<T: m::Alias>() { ... }
```
(Trait objects are considered types, so they are covered by previous
paragraphs.)

## Private-in-public lints

Previously type privacy was ensured by so called private-in-public errors,
that worked preventively.
```rust
mod m {
    struct Priv;

    // ERROR, private type `Priv` in public interface.
    pub fn leak() -> Priv { ... }
}

// Can't obtain a value of `Priv` because for `leak` the function definition
// itself is illegal.
let x = m::leak();
```

The logic behind private-in-public rules is very simple, if some type has
visibility `vis_type` then it cannot be used in interfaces of items with
visibilities `vis_interface` where `vis_interface > vis_type`.  
In particular, this code is illegal
```rust
mod outer {
    struct S;

    mod inner {
        pub fn f() -> S { ... }
    }
}
```
for a simple reason -
`vis(f) = pub, vis(S) = pub(in outer), pub > pub(in outer)`.
Many people found this confusing because they expected private-in-public rules
to be based on crate-global reachability and not on local `pub` annotations.  
(Both `S` and `f` are reachable only from `outer` despite `f` being `pub`.)

In addition, private-in-public rules were found to be
[insufficient](https://github.com/rust-lang/rust/issues/30476)
for ensuring type privacy due to type inference being quite smart.  
As a result, type privacy checking was implemented directly - when we see value
`m::leak()` we just check if its type private or not, so private-in-public
rules became not-strictly-necessary for the compiler.

However, private-in-public diagnostics are still pretty useful for humans!  
For example, if a function is defined like this
```
mod m {
    struct Priv;
    pub fn f() -> Priv { ... }
}
```
it's *guaranteed* to be unusable outside of `m` because every its use will cause
a type privacy error.  
That's probably not what the author of `f` wanted. Either `Priv` is supposed to
be public, or `f` is supposed to be private. It would be nice to diagnose
cases like this, but to avoid "false positives" like the previous example with
`outer`/`inner`.  
Meet reachability-based private-in-public *lints*!

### Lint #1: Private types in primary interface of effectively public items

Effective visibility of an item is how far it's actually reexported or leaked
through other means, like return types.  
Effective visibility can never be larger than nominal visibility (i.e. what
`pub` annotation says), but it can be smaller.

For example, in the `outer`/`inner` example nominal visibility of `f` is `pub`,
but its effective visibility is `pub(in outer)`, because it's neither reexported
from `outer`, nor can be named directly from outside of it.  
`effective_vis(f) <= vis(Priv)` means that the private-in-public lint #1 is
*not* reported for `f`.

"Primary interface" in the lint name means everything in the interface except
for trait bounds and `where` clauses, those are considered secondary interface.
```rust
trait PrivTr {}
pub fn bad()
    -> Box<PrivTr> // WARN, private type in primary interface
{ ... }
pub fn better<T>(arg: T)
    where T: PrivTr // OK, private trait in secondary interface
{ ... }
```
This lint replaces part of private-in-public errors. Having something
private in primary interface guarantees that the item will be unusable from
outer modules due to type privacy (primary interface is considered part of the
type when type privacy is checked), so it's very desirable to warn about this
situation in advance and this lint needs to be at least warn-by-default.

Provisional name for the lint - `private_interfaces`.

### Lint #2: Private traits/types in secondary interface of effectively public items

This lint is reported if private types or traits are found in trait bounds or
`where` clauses of an effectively public item.
```rust
trait PrivTr {}
pub fn overloaded<T>(arg: T)
    where T: PrivTr // WARN, private trait in secondary interface
{ ... }
```
Function `overloaded` has public type, can't leak values of any other private
types and can be freely used outside of its module without causing type privacy
errors. There are reasonable use cases for such functions, for example emulation
of sealed traits.  
The only suspicious part about it is documentation - what arguments can it take
exactly? The set of possible argument types is closed and determined by
implementations of the private trait `PrivTr`, so it's kinda mystery unless it's
well documented by the author of `overloaded`.  
There are stability implications as well - set of possible `T`s is still an
interface of `overload`, so impls of `PrivTr` cannot be removed
backward-compatibly.  
This lint replaces part of private-in-public errors and can be reported as
warn-by-default or allow-by-default.

Provisional name for the lint - `private_bounds`.

### Lint #3: "Voldemort types" (it's reachable, but I can't name it)

Consider this code
```rust
mod m {
    // `S` has public nominal and effective visibility,
    // but it can't be *named* outside of `m::super`.
    pub struct S;
}

// OK, can return public type `m::S` and
// can use the returned value in outer modules.
// BUT, we can't name the returned type, unless we have `typeof`,
// and we don't have it yet.
pub fn get_voldemort() -> m::S { ... }
```
The "Voldemort type" (or, more often, "Voldemort trait") pattern has legitimate
uses, but often it's just an oversight and `S` is supposed to be reexported and
nameable from outer modules.  
The lint is supposed to report items for which effective visibility is larger
than the area in which they can be named.  
This lint is new and doesn't replace private-in-public errors, but it provides
checking that many people *expected* from private-in-public.  
The lint should be allow-by-default or it can be placed into Clippy as an
alternative.

Provisional name for the lint - `unnameable_types`.

### Lint #4: `private_in_public`

Some private-in-public erros are currently reported as a lint
`private_in_public` for compatibility reasons.  
This compatibility lint will be removed and its uses will be reported as
warnings by `renamed_and_removed_lints`.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Type privacy

### How to determine visibility of a type?

- Built-in types are considered `pub` (integer and floating point types, `bool`,
`char`, `str`, `!`).
- Type parameters (including `Self` in traits) are considered `pub` as well.
- Arrays and slices inherit visibility from their element types.  
`vis([T; N]) = vis([T]) = vis(T)`.
- References and pointers inherit visibility from their pointee types.  
`vis(&MUTABILITY T) = vis(*MUTABILITY T) = vis(T)`.
- Tuple types are as visible as their least visible component.  
`vis((A, B)) = min(vis(A), vis(B))`.
- Struct, union and enum types are as visible as their least visible type
argument or type constructor.  
`vis(Struct<A, B>) = min(vis(Struct), vis(A), vis(B))`.
- Closures and generators have same visibilities as equivalent structs defined
in the same module.  
`vis(CLOSURE<A, B>) = min(vis(CURRENT_MOD), vis(A), vis(B))`.
- Traits or trait types are as visible as their least visible type
argument or trait constructor.  
`vis(Tr<A, B>) = min(vis(Tr), vis(A), vis(B))`.
- Trait objects and `impl Trait` types are as visible as their least visible
component.  
`vis(TrA + TrB) = vis(impl TrA + TrB) = min(vis(TrA), vis(TrB))`.
- Non-normalizable associated types are as visible as their least visible
component.  
`vis(<Type as Trait>::AssocType) = min(vis(Type), vis(Trait))`.
- Function pointer types are as visible as least visible types in their
signatures.  
`vis(fn(A, B) -> R) = min(vis(A), vis(B), vis(R))`.
- Function item types are as visible as their least visible component as well,
but the definition of a "component" is a bit more complex.
    - For free functions and foreign functions components include signature,
    type parameters and the function item's nominal visibility.  
    `vis(fn(A, B) -> R { foo<C> }) = min(vis(fn(A, B) -> R), vis(C), vis(foo))`
    - For struct and enum variant constructors components include signature,
    type parameters and the constructor item's nominal visibility.  
    `vis(fn(A, B) -> S<C> { S_CTOR<C> }) = min(vis(fn(A, B) -> S<C>), vis(S_CTOR))`.  
    `vis(fn(A, B) -> E<C> { E::V_CTOR<C> }) = min(vis(fn(A, B) -> E<C>), vis(E::V_CTOR))`.  
    `vis(S_CTOR) = min(vis(S), vis(field_1), ..., vis(field_N))`.  
    `vis(E::V_CTOR) = vis(E)`.
    - For inherent methods components include signature, impl type, type
    parameters and the method's nominal visibility.  
    `vis(fn(A, B) -> R { <Type>::foo<C> })) = min(vis(fn(A, B) -> R), vis(C), vis(Type), vis(foo))`.
    - For trait methods components include signature, trait, type parameters
    (including impl type `Self`) and the method item's nominal visibility
    (inherited from the trait, included automatically).  
    `vis(fn(A, B) -> R { <Type as Trait>::foo<C> })) = min(vis(fn(A, B) -> R), vis(C), vis(Type), vis(Trait))`.
- "Infer me" types `_` are replaced with their inferred types before checking.

### The type privacy rule

A type or a trait private to module `m` (`vis(in m)`) cannot be used outside of
that module (`vis(outside) > vis(in m)`).  
Uses include naming this type or trait (possibly through aliases) or obtaining
values (expressions or patterns) of this type.

The rule is enforced non-hygienically.  
So it's possible for a macro 2.0 to name some private type without causing name
privacy errors, but it will still be reported as a type privacy violation.  
This can be partially relaxed in the future, but such relaxations are out of
scope for this RFC.

### Additional restrictions for associated items

For technical reasons it's not always desirable or possible to fully normalize
associated types before checking them for privacy.  
So, if we see `<Type as Trait>::AssocType` we can guaranteedly check only `Type`
and `Trait`, but not the resulting type.  
So we must be sure it's no more private than what we can check.

As a result, private-in-public violations for associated type definitions
are still eagerly reported as errors, using the old rules based on local `pub`
annotations and not reachability.
```rust
struct Priv;
pub struct Type;
pub trait Trait {}

impl Trait for Type {
    type AssocType = Priv; // ERROR, vis(Priv) < min(vis(Trait), vis(Type))
}
```

When associated function is defined in a private impl (i.e. the impl type or
trait is private) it's guaranteed that the function can't be used outside of
the impl's area of visibility.  
Type privacy ensures this because associated functions have their own unique
types attached to them.  

Associated constants and associated types from private impls don't have attached
unique types, so they sometimes can be used from outer modules due to
sufficiently smart type inference.
```rust
mod m {
    struct Priv;
    pub struct Pub<T>(T);
    pub Trait { type A; }

    // This is a private impl because `Pub<Priv>` is a private type
    impl Pub<Priv> {
        const C: u8 = 0;
    }

    // This is a private impl because `Pub<Priv>` is a private type
    impl Trait for Pub<Priv> { type A = u8; }
}
use m::*;

// But we still can use `C` outside of `m`?
let x = Pub::C; // With type inference this means `<Pub<Priv>>::C`
```

It would be good to provide the same guarantees for associated constants
and types as for associated functions.  
As a result, type privacy additionally prohibits use of any associated items
from private impls.
```rust
// ERROR, `C` is from a private impl with type `Pub<Priv>`
let x = Pub::C;
// ERROR, `A` is from a private impl with type `Pub<Priv>`,
// even if the whole type of `x` is public `u8`.
let x: <Pub<_> as Trait>::A;
```
In principle, this restriction can be considered a part of the primary type
privacy rule - "can't name a private type" - if all `_`s (types to infer, 
explicit or implicit) are replaced by their inferred types before checking, so
`Pub` and `Pub<_>` in the examples above become `Pub<Priv>`.

### Lints

Effective visibility of an item is determined by a module into which it can be
leaked through
- chain of public parent modules (they make it directly nameable)
- chains of reexports or type aliases (they make it nameable through aliases)
- functions, constants, fields "returning" the value of this item, if the item
is a type
- maybe something else if deemed necessary, but probably not macros 2.0.

(Here we consider the "whole universe" a module too for uniformity.)  
If effective visibility of an item is larger than its nominal visibility
(`pub` annotation), then it's capped by the nominal visibility.

Primary interface of an item is all its interface (types of returned values,
types of fields, types of fn parameters) except for bounds on generic parameters
and `where` clauses.

Secondary interface of an item consists of bounds on generic parameters and
`where` clauses, including supertraits for trait items.

Lint `private_interfaces` is reported when a type with visibility `x` is used
in primary interface of an item with effective visibility `y` and `x < y`.  
This lint is warn-by-default.

Lint `private_bounds` is reported when a type or trait with visibility `x` is
used in secondary interface of an item with effective visibility `y` and
`x < y`.  
This lint is warn-by-default.

Lint `unnameable_types` is reported when effective visibility of a type is
larger than module in which it can be named, either directly, or through
reexports, or through trivial type aliases (`type X = Y;`, no generics on both
sides).  
This lint is allow-by-default.

Compatibility lint `private_in_public` is never reported and removed.

# Drawbacks
[drawbacks]: #drawbacks

With
```rust
pub fn f<T>(arg: T)
    where T: PrivateTrait
{ ... }
```
being legal (even if it's warned against by default) the set of
`PrivateTrait`'s implementations becomes a part of `f`'s interface.
`PrivateTrait` can still be freely renamed or even splitted into several traits
though.  
`rustdoc` may be not fully prepared to document items with private traits in
bounds, manually written documentation explaining how to use the interface
may be required.

# Rationale and Alternatives
[alternatives]: #alternatives

Names for the lints are subject to bikeshedding.

`private_interfaces` and `private_bounds` can be merged into one lint.
The rationale for keeping them separate is different probabilities
of errors in case of lint violations.  
The first lint indicates an almost guaranteed error on client side,
the second one is more in the "missing documentation" category.

# Unresolved questions
[unresolved]: #unresolved-questions

It's not fully clear if the restriction for associated type definitions required for
type privacy soundness, or it's just a workaround for a technical difficulty.

Interactions between macros 2.0 and the notions of reachability / effective
visibility used for the lints are unclear.
