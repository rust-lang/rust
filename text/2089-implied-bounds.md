- Feature Name: `implied_bounds`
- Start Date: 2017-07-28
- RFC PR: https://github.com/rust-lang/rfcs/pull/2089
- Rust Issue: https://github.com/rust-lang/rust/issues/44491

# Summary
[summary]: #summary

Eliminate the need for “redundant” bounds on functions and impls where those bounds can be inferred from the input types and other trait bounds. For example, in this simple program, the impl would no longer require a bound, because it can be inferred from the `Foo<T>` type:

```rust
struct Foo<T: Debug> { .. }
impl<T: Debug> Foo<T> {
  //    ^^^^^ this bound is redundant
  ...
}
```
Hence, simply writing `impl<T> Foo<T> { ... }` would suffice. We currently support implied bounds for lifetime bounds, super traits and projections. We propose to extend this to all where clauses on traits and types, as was already discussed [here][niko].

# Motivation
[motivation]: #motivation

## Types

Let's take an example from the standard library where trait bounds are actually expressed on a type¹.
```rust
pub enum Cow<'a, B: ?Sized + 'a>
    where B: ToOwned
{
    Borrowed(&'a B),
    Owned(<B as ToOwned>::Owned),
}
```
The `ToOwned` bound has then to be carried everywhere:
```rust
impl<'a, B: ?Sized> Cow<'a, B>
    where B: ToOwned
{
    ...
}

impl<'a, B: ?Sized> Clone for Cow<'a, B>
    where B: ToOwned
{
    ...
}

impl<'a, B: ?Sized> Eq for Cow<'a, B: Eq>
    where B: ToOwned
{
    ...
}
```
even if one does not actually care about the semantics implied by `ToOwned`:
```rust
    fn panic_if_not_borrowed<'a, B>(cow: Cow<'a, B>) -> &'a B
//      where B: ToOwned
    {
        match cow {
            Cow::Borrowed(b) => b,
            Cow::Owned(_) => panic!(),
        }
    }
//  ^ the trait `std::borrow::ToOwned` is not implemented for `B`
```
However what we know is that if `Cow<'a, B>` is well-formed, then `B` *has* to implement `ToOwned`. We would say that such a bound is *implied* by the well-formedness of `Cow<'a, B>`.

Currently, impls and functions have to prove that their arguments are well-formed. Under this proposal, they would *assume* that their arguments are well-formed, leaving the responsibility for proving well-formedness to the caller. Hence we would be able to drop the `B: ToOwned` bounds in the previous examples.

Beside reducing repeated constraints, it would also provide a clearer separation between what bounds a type needs so that it is well-formed, and what additional bounds an `fn` or an `impl` actually needs:

```rust
struct Set<K> where K: Hash + Eq { ... }

fn only_clonable_set<K: Hash + Eq + Clone>(set: Set<K>) { ... }

// VS

fn only_clonable_set<K: Clone>(set: Set<K>) { ... }
```

Moreover, we already support implied lifetime bounds on types:
```rust
pub struct DebugStruct<'a, 'b> where 'b: 'a {
    fmt: &'a mut fmt::Formatter<'b>,
    ...
}

pub fn debug_struct_new<'a, 'b>(fmt: &'a mut fmt::Formatter<'b>, name: &str) -> DebugStruct<'a, 'b>
//  where 'b: 'a
//  ^^^^^^^^^^^^  this is not needed
{
    /* inside here: assume that `'b: 'a` */
}
```
This RFC proposes to extend this sort of logic beyond these special cases and use it uniformly for both trait bounds and lifetime bounds.

¹Actually only a few types in the standard library have bounds, for example `HashSet<T>` does not have a `T: Hash + Eq` on the type declaration, but on the impl declaration rather. Whether we should prefer bounds on types or on impls is related, but beyond the scope of this RFC.

## Traits

Traits also currently support some form of implied bounds, namely super traits bounds:
```rust
// Equivalent to `trait Foo where Self: From<Bar>`.
trait Foo: From<Bar> { }

pub fn from_bar<T: Foo>(bar: Bar) -> T {
    // `T: From<Bar>` is implied by `T: Foo`.
    T::from(bar)
}
```
and bounds on projections:
```rust
// Equivalent to `trait Foo where Self::Item: Eq`.
trait Foo {
    type Item: Eq;
}

fn only_eq<T: Eq>() { }

fn foo<T: Foo>() {
    // `T::Item: Eq` is implied by `T: Foo`.
    only_eq::<T::Item>()
}
```
However, this example does not compile:
```rust
    trait Foo<U> where U: Eq { }

    fn only_eq<U: Eq>() { }

    fn foo<U, T: Foo<U>>() {
        only_eq::<U>()
    }
//  ^ the trait `std::cmp::Eq` is not implemented for `U`
```
Again we propose to uniformly support implied bounds for all where clauses on trait definitions.

# Guide-Level Explanation
[guide]: #guide

When you declare bounds on a type, you don't have to repeat them when writing impls and functions as soon as the type appear in the signature or the impl header:
```rust
struct Set<T> where T: Hash + Eq {
    ...
}

impl<T> Set<T> {
    // You can rely on the fact that `T: Hash + Eq` inside here.
    ...
}

impl<T> Clone for Set<T> where T: Clone {
    // Same here, and you can also rely on the `T: Clone` bound of course.
    ...
}

fn only_eq<U: Eq>() { }

fn use_my_set<T>(arg: Set<T>) {
    // We know that `T: Eq` because we have a `Set<T>` as an argument, and there already is a
    // `T: Eq` bound on the declaration of `Set`.
    only_eq::<T>();
}

// This also works for the return type: no need to repeat bounds.
fn return_a_set<T>() -> Set<T> {
    Set::new()
}
```

Lifetime bounds are supported as well (this is already the case today):
```rust
struct MyStruct<'a, 'b> where 'b: 'a {
    reference: &'a &'b i32,
}

fn use_my_struct<'a, 'b>(arg: MyStruct<'a, 'b>) {
    // No need to repeat `where 'b: 'a`, it is assumed.
}
```

However, you still have to write the bounds explicitly if the type does not appear in the function signature or the impl header:
```rust
// `Set<T>` does not appear in the fn signature: we need to explicitly write the bounds.
fn declare_a_set<T: Hash + Eq>() {
    let set = Set::<T>::new();
}
```

Similarly, you don't have to repeat bounds that you write on a trait declaration as soon as you know that the trait reference holds:
```rust
trait Foo where Bar: Into<Self> {
    ...
}

fn into_foo<T: Foo>(bar: Bar) -> T {
    // We know that `T: Foo` holds so given the trait declaration, we know that `Bar: Into<T>`.
    bar.into()
}
```

Note that this is transitive:
```rust
trait Foo { }
trait Bar where Self: Foo { }
trait Baz where Self: Bar { }

fn only_foo<T: Foo>() { }

fn use_baz<T: Baz>() {
    // We know that `T: Baz`, hence we know that `T: Bar`, hence we know that `T: Foo`.
    only_foo::<T>()
}
```

This also works for bounds on associated types:
```rust
trait Foo {
    type Item: Debug;
}

fn debug_foo<U, T: Foo<Item = U>>(arg: U) {
    // We know that `<T as Foo>::Item` implements `Debug` because of the trait declaration.
    // Moreover, we know that `<T as Foo>::Item` is `U`.
    // Hence, we know that `U` implements `Debug`.
    println!("{:?}", arg);

    /* do something else with `T` and `U`... */
}
```

# Reference-Level Explanation
[reference]: #reference

This is the fully-detailed design and you probably don't need to read everything. This design has already been experimented on [Chalk](https://github.com/nikomatsakis/chalk), to some extent. The current design has been driven by issue [#12], it is a good read to understand why we *need* to expand where clauses as described below.

We'll use the grammar from [RFC 1214] to detail the rules:
```
T = scalar (i32, u32, ...)              // Boring stuff
  | X                                   // Type variable
  | Id<P0, ..., Pn>                     // Nominal type (struct, enum)
  | &r T                                // Reference (mut doesn't matter here)
  | O0 + ... + On + r                   // Object type
  | [T]                                 // Slice type
  | for<r...> fn(T1, ..., Tn) -> T0     // Function pointer
  | <P0 as Trait<P1, ..., Pn>>::Id      // Projection
P = r                                   // Region name
  | T                                   // Type
O = for<r...> TraitId<P1, ..., Pn>      // Object type fragment
r = 'x                                  // Region name
```

We'll use the same notations as [RFC 1214] for the set `R = <r0, ..., rn>` denoting the set of lifetimes currently bound.

## Well-formedness rules
Basically, we say that something (type or trait reference) is well-formed if the bounds declared on it are met, *regardless of the well-formedness of its parameters*: this is the main difference with [RFC 1214].

We will write:
* `WF(T: Trait)` for a trait reference `T: Trait` being well-formed
* `WF(T)` for a reference to the type `T` being well-formed

### **Trait refs**
We'll start with well-formedness for trait references. The important thing is that we distinguish between `T: Trait` and `WF(T: Trait)`. The former means that an impl for `T` has been found while the latter means that `T` meets the bounds on trait `Trait`.

We'll also consider a function `Expanded` applying on where clauses like this:
```
Expanded((T: Trait)) = { (T: Trait), WF(T: Trait) }
Expanded((T: Trait<Item = U>)) = { (T: Trait<Item = U>), WF(T: Trait) }
Expanded(OtherWhereClause) = { OtherWhereClause }
```
We naturally extend `Expanded` so that it applies on a finite set of where clauses:
```
Expanded({ WhereClause1, ..., WhereClauseN }) = Union(Expanded(WhereClause1), ..., Expanded(WhereClauseN))
```
***Every where clause*** a user writes will be expanded through the `Expanded` function. This means that the following impl:
```rust
impl<T, U> Into<T> for U where T: From<U> { ... }
```
will give the following rule:
```
 T: From<U>, WF(T: From<U>)
--------------------------------------------------
 U: Into<T>
```

Now let's see the actual rule for a trait reference being well-formed:
```
WfTraitReference:
  C = Expanded(WhereClauses(TraitId))   // the conditions declared on TraitId must hold...
  R, r... ⊢ [P0, ..., Pn] C             // ...after substituting parameters, of course
  --------------------------------------------------
  R ⊢ WF(for<r...> P0: TraitId<P1, ..., Pn>)
```

 And here is an example:
```rust
// `WF(Self: SuperTrait)` holds.
trait SuperTrait { }

// `WF(Self: Trait)` holds if `Self: SuperTrait`, `WF(Self: Supertrait)`.
trait Trait: SuperTrait { }

// `i32: Trait` holds but not `WF(i32: Trait)`.
// This would be flagged as an error.
impl Trait for i32 { }

// Both `f32: Trait` and `WF(f32: Trait)` hold.
impl SuperTrait for f32 { }
impl Trait for f32 { }
```

### **Types**

The well-formedness rules for types are given by:
```
WfScalar:
  --------------------------------------------------
  R ⊢ WF(scalar)

WfFn:                              // an fn pointer is always WF since it only carries parameters
  --------------------------------------------------
  R ⊢ WF(for<r...> fn(T1, ..., Tn) -> T0)

WfObject:
  rᵢ = union of implied region bounds from Oi
  ∀i. rᵢ: r
  --------------------------------------------------
  R ⊢ WF(O0 + ... + On + r)

WfObjectFragment:
  TraitId is object safe
  --------------------------------------------------
  R ⊢ WF(for<r...> TraitId<P1, ..., Pn>)

WfTuple:
  ∀i<n. R ⊢ Ti: Sized              // the *last* field may be unsized
  --------------------------------------------------
  R ⊢ WF((T1, ... ,Tn))

WfNominalType:
  C = Expanded(WhereClauses(Id))   // the conditions declared on Id must hold...
  R ⊢ [P1, ..., Pn] C              // ...after substituting parameters, of course
  --------------------------------------------------
  R ⊢ WF(Id<P1, ..., Pn>)

WfReference:
  R ⊢ T: 'x                        // T must outlive 'x
  --------------------------------------------------
  R ⊢ WF(&'x T)

WfSlice:
  R ⊢ T: Sized
  --------------------------------------------------
  R ⊢ WF([T])

WfProjection:
  R ⊢ P0: Trait<P1, ..., Pn>       // the trait reference holds
  R ⊢ WF(P0: Trait<P1, ..., Pn>)   // the trait reference is well-formed
  --------------------------------------------------
  R ⊢ WF(<P0 as Trait<P1, ..., Pn>>::Id)
```
Taking again our `SuperTrait` and `Trait` from above, here is an example:
```rust
// `WF(Struct<T>)` holds if `T: Trait`, `WF(T: Trait)`.
struct Struct<T> where T: Trait {
    field: T,
}

// `WF(Struct<i32>)` would not hold since `WF(i32: Trait)` doesn't.
// But `WF(Struct<f32>)` does hold.
```

## Reverse rules
This is a core element of this RFC. Morally, the well-formedness rules are "if and only if" rules. We thus add reverse rules for each relevant WF rule:
```
ReverseWfTraitReferenceᵢ
  // Substitute parameters
  { WhereClause1, ..., WhereClauseN } = [P0, ..., Pn] Expanded(WhereClauses(TraitId))
  R ⊢ WF(for<r...> P0: TraitId<P1, ..., Pn>)
  --------------------------------------------------
  R, r... ⊢ WhereClauseᵢ

ReverseWfTupleᵢ, i < n:
  R ⊢ WF((T1, ..., Tn))
  --------------------------------------------------
  R ⊢ Ti: Sized   // not very useful since this bound is often implicit

ReverseWfNominalTypeᵢ:
  // Substitute parameters
  { WhereClause1, ..., WhereClauseN } = [P1, ..., Pn] Expanded(WhereClauses(id))
  R ⊢ WF(Id<P1, ..., Pn>)
  --------------------------------------------------
  R ⊢ WhereClauseᵢ

ReverseWfReference:
  R ⊢ WF(&'x T)
  --------------------------------------------------
  R ⊢ T: 'x

ReverseWfSlice:
  R ⊢ WF([T])
  --------------------------------------------------
  R ⊢ T: Sized    // same as above
```

Note that we add reverse rules for all ***expanded*** where clauses, this means that given:
```rust
// Expands to `trait Foo where Self: Bar, WF(Self: Bar)`
trait Bar where Self: Foo { }
```
we have two reverse rules given by:
```
WF(T: Bar)
--------------------------------------------------
T: Foo

WF(T: Bar)
--------------------------------------------------
WF(T: Foo)
```

**Remark**: Reverse rules include implicit `Sized` bounds on type declarations. However, they do not include (explicit) `?Sized` bounds since those are not *real* trait bounds, but only a way to disable the implicit `Sized` bound.

## Input types
We define the notion of input types of a type. Basically, input types refer to all types that are accessible from referencing to a specific type. For example, a function will assume that the input types of its arguments are well-formed, hence in the body of that function we'll be able to derive implied bounds thanks to the reverse rules described earlier.

We'll denote by `InputTypes` the function which maps a type to its input types, defined by:
```
// Scalar
InputTypes(scalar) = { scalar }

// Type variable
InputTypes(X) = { X }

// Region name
InputTypes(r) = { }

// Reference
InputTypes(&r T) = Union({ &r T }, InputTypes(T))

// Slice type
InputTypes([T]) = Union({ [T] }, InputTypes(T))

// Nominal type
InputTypes(Id<P0, ..., Pn>) = Union({ Id<P0, ..., Pn> }, InputTypes(P0), ..., InputTypes(Pn))

// Object type
InputTypes(O0 + ... + On + r) = Union({ O0 + ... + On + r }, InputTypes(O0), ..., InputTypes(On))

// Object type fragment
InputTypes(for<r...> TraitId<P1, ..., Pn>) = { for<r...> TraitId<P1, ..., Pn> }

// Function pointer
InputTypes(for<r...> fn(T1, ..., Tn) -> T0) = { for<r...> fn(T1, ..., Tn) -> T0 }

// Projection
InputTypes(<P0 as Trait<P1, ..., Pn>>::Id) = Union(
    { <P0 as Trait<P1, ..., Pn>>::Id },
    InputTypes(P0),
    InputTypes(P1),
    ...,
    InputTypes(Pn)
)
```

Note that higher-ranked types (functions, object type fragments) do not carry input types other than themselves. This is because they are unusable *as such*, one will have to use them in a lower-ranked way at some point (e.g. calling a function) and will thus rely on `InputTypes` for normal types.

## Assumptions and checking well-formedness
This is the other core element: how to use reverse rules. Basically, functions and impls will assume that their input types are well-formed, and that (expanded) where clauses hold.

### **Functions**
Given a function declaration:
```rust
fn F<r..., X1, ..., Xn>(arg1: T1, ..., argm: Tm) -> T0 where WhereClause1, ..., WhereClausek {
    /* body of the function inside here */
}
```
We rely on the following assumptions inside the body of `F`:
* `Expanded({ WhereClause1, ..., WhereClausek })`
* `WF(T)` for all `T ∈ Union(InputTypes(T0), InputTypes(T1), ..., InputTypes(Tm))`
* `WF(Xi)` for all `i`

Note that we assume that the input types of the return type `T0` are well-formed.

With these assumptions, the function must be able to prove that everything that appears in its body is well-formed (e.g. every type appearing in the body, projections, etc).

Moreover, a caller of `F` would have to prove that the where clauses on `F` hold, after having substituted parameters.

**Remark**: Notice that we assume that the type variables `Xi` are well-formed for all `i`. This way, type variables don't need a special treatment regarding well-formedness. See example below.

Examples:

```rust
trait Bar { }
trait Foo where Box<Self>: Bar { }

fn only_bar<T: Bar>() { }

fn foo<T: Foo>() {
    // Inside the body, we have to prove `WF(T)`, `WF(Box<T>)`, and `Box<T>: Bar`.
    // Because we assume that `WF(T: Foo)`, we indeed have `Box<T>: Bar`.
    only_bar::<Box<T>>()
}

fn main() {
    // We have to prove `WF(i32)`, `i32: Foo`.
    foo::<i32>();
}
```

```rust
/// Illustrate remark 2: no need for a special treatment for type variables.

struct Set<K: Hash> { ... }

fn two_variables<T, U>() { }

fn one_variable<T: Hash>() {
    // We have to prove `WF(T)`, `WF(Set<T>)`. `WF(T)` trivially holds because of the assumption
    // made by the function `one_variable`. `WF(Set<T>)` holds because of the `T: Hash` bound.
    two_variables<T, Set<T>>()
}

fn main() {
    // We have to prove `WF(i32)`.
    one_variable::<i32>();
}
```

```rust
/// Illustrate "inner" input types and transitivity

trait Bar where Box<Self>: Eq { }
trait Baz: Bar { }

struct Struct<T: Baz> { ... }

fn only_eq<T: Eq>() { }

fn dummy<T>(arg: Option<Struct<T>>) {
    /* do something with arg */

    // Since `Struct<T>` is an input type, we assume that `WF(Struct<T>)` hence `WF(T: Baz)`
    // hence `WF(T: Bar)` hence `Box<T>: Eq`
    only_eq::<Box<T>>()
}
```

### **Trait impls**
Given a trait impl:
```rust
impl<r..., X1, ..., Xn> Trait<r'..., T1, ..., Tn> for T0 where WhereClause1, ..., WhereClausek {
    // body of the impl inside here

    type Assoc = AssocTyValue;

    /* ... */
}
```
We rely on the following assumptions inside the body of the impl:
* `Expanded({ WhereClause1, ..., WhereClausek })`
* `WF(T)` for all `T ∈ Union(InputTypes(T0), InputTypes(T1), ..., InputTypes(Tn))`
* `WF(Xi)` for all `i`

Based on these assumptions, the impl declaration has to prove `WF(T0: Trait<r'..., T1, ..., Tn>)` and `WF(T)` for all `T ∈ InputTypes(AssocTyValue)`. Note that associated fns can be seen as (higher-kinded) associated types, but since fn pointers are always well-formed and do not carry input types other than themselves, this is fine.

Associated fns make their normal assumptions + the set of assumptions made by the impl. Things to prove inside associated fns do not differ from normal fns.

Note that when projecting out of a type, one must automatically prove that the trait reference holds because of the `WfProjection` rule.

Examples:

```rust
struct Set<K: Hash> { ... }

trait Foo where Self: Clone {
    fn foo();
}

fn only_hash<T: Hash>() { }

impl<K: Clone> Foo for Set<K> {
    // Inside here: we assume `WF(Set<K>)`, `K: Clone`, `WF(K: Clone)`, `WF(K)`.
    // Also, we must prove `WF(Set<K>: Foo)`.

    fn foo() {
        only_hash::<K>()
    }
}
```

```rust
struct Set<K: Hash> { ... }

trait Foo {
    type Item;
}

// We need an explicit `K: Hash` bound in order to prove that the associated type value `Set<K>` is WF.
impl<K: Hash> Foo for K {
    type Item = Set<K>;
}
```

```rust
trait Foo {
    type Item;
}

impl<T> Foo for T where T: Clone {
    type Item = f32;
}

fn foo<T: Foo>(arg: T) {
    // We must prove `WF(<T as Foo>::Item)` hence prove that `T: Foo`: ok this is in our assumptions.
    let a = <T as Foo>::Item;
}

fn bar<T: Clone>(arg: T) {
    // We must prove `WF(<T as Foo>::Item)` hence prove that `T: Foo`: ok, use the impl.
    let a = <T as Foo>::Item;
}
```

### **Inherent impls**
Given an inherent impl:
```rust
impl<r..., X1, ..., Xn> SelfTy where WhereClause1, ..., WhereClausek {
    /* body of the impl inside here */
}
```
We rely on the following assumptions inside the body of the impl:
* `Expanded({ WhereClause1, ..., WhereClausek })`
* `WF(T)` for all `T ∈ InputTypes(SelfTy)`
* `WF(Xi)` for all `i`

Methods make their normal assumptions + the set of assumptions made by the impl. Things to prove inside methods do not differ from normal fns.

A caller of a method has to prove that the where clauses defined on the impl hold, in addition to the requirements for calling general fns.

## Proving well-formedness for input types
[proving-wf-input-types]: #proving-well-formedness-for-input-types

One would have noticed that we only prove well-formedness for input types in a lazy way (e.g., inside function bodies). This means that if we have a function:
```rust
struct Set<K: Hash> { ... }
struct NotHash;

fn foo(arg: Set<NotHash>) { ... }
```
then no error will be caught until someone actually tries to call `foo`. Same thing for an impl:
```rust
impl Set<NotHash> { ... }
```
the error will not be caught until someone actually uses `Set<NotHash>`.

The idea is, when encountering an fn/trait impl/inherent impl, retrieve all input types that appear in the signature / header and for each input type `T`, do the following: retrieve type variables `X1, ..., Xn` bound by the declaration and ask for `∃X1, ..., ∃Xn; WF(T)` in an empty enviromnent (in Chalk terms). If there is no possible substitution for the existentials, output a warning.

Example:
```rust
struct Set<K: Hash> { ... }

// `NotHash` is local to this crate, so we know that there exists no `T`
// such that `NotHash<T>: Hash`.
struct NotHash<T> { ... }

// Warning: `foo` cannot be called whatever the value of `T`
fn foo<T>(arg: Set<NotHash<T>>) { ... }
```

## Cycle detection
In Chalk this design often leads to cycles in the proof tree. Example:
```rust
trait Foo { }
// `WF(Self: Foo)` holds.
 
impl Foo for u8 { }

// Expanded to `trait Bar where Self: Foo, WF(Self: Foo)`
trait Bar where Self: Foo { }

// WF rule:
// `WF(Self: Bar)` holds if `Self: Foo`, `WF(Self: Foo)`.

// Reverse WF rules:
// `Self: Foo` holds if `WF(Self: Bar)`
// `WF(Self: Foo)` holds if `WF(Self: Bar)`
```
Now suppose we are asking wether `u8: Foo` holds. The following branch exists in the proof tree:
`u8: Foo` holds if `WF(u8: Bar)` holds if `u8: Foo` holds.

I *think* rustc would have the right behavior currently: just dismiss this branch since it only leads to the tautological rule `(u8: Foo) if (u8: Foo)`.

In Chalk we have a more sophisticated cycle detection strategy based on tabling, which basically enables us to correctly answer "multiple solutions", instead of "unique solution" if a simple *error-on-cycle* strategy were used. Would rustc need such a thing?

# Drawbacks
[drawbacks]: #drawbacks

* Implied bounds on types can feel like "implicit bounds" (although they are not: the types appear in the signature of a function / impl header, so it's self-documenting).
* Removing a bound from a struct becomes a breaking change (note: this can already be the case for functions and traits).

# Rationale and Alternatives
[alternatives]: #alternatives

## Including parameters in well-formedness rules

Specific to this design: instead of disregarding parameters in well-formedness checks, we could have included them, and added reverse rules of the form: "`WF(T)` holds if `WF(Struct<T>)` holds". From a theoretical point of view, this would have had the same effects as the current design, and would have avoided the whole `InputTypes` thing. However, implementation in Chalk revelead some tricky issues. Writing in Chalk-style, suppose we have rules like:
```
WF(Struct<T>) :- WF(T)
WF(T) :- WF(Struct<T>)
```
then trying to prove `WF(i32)` gives birth to an infinite branch `WF(i32) :- WF(Struct<i32>) :- WF(Struct<Struct<i32>>) :- ...` in the proof tree, which is hard (at least that's what we believe) to dismiss.

## Trait aliases

Trait aliases offer a way to factorize repeated constraints ([RFC 1733]), it's useful especially for bounds on types, but it does not overcome the limitations for implied bounds on traits (the `where Bar: Into<Self>` example is a good one).

## Limiting the scope of implied bounds

These essentially try to address the breaking change when removing a bound on a type:
* do not derive implied bounds for types
* limit the use of implied bounds for types that are in your current crate only
* derive implied bounds in impl bodys only
* two distinct feature-gates, one for implied bounds on traits and another one for types

# Unresolved questions
[unresolved]: #unresolved-questions

* Should we try to limit the range of implied bounds to be crate-local (or module-local, etc)?
* @nikomatsakis pointed [here][niko] that implied bounds can interact badly with current inference rules.

[#12]: https://github.com/nikomatsakis/chalk/issues/12

[RFC 1214]: https://github.com/rust-lang/rfcs/blob/master/text/1214-projections-lifetimes-and-wf.md

[RFC 1733]: https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md

[niko]: https://internals.rust-lang.org/t/lang-team-minutes-implied-bounds/4905
