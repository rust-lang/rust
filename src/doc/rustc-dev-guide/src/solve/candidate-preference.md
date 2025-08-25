# Candidate preference

There are multiple ways to prove `Trait` and `NormalizesTo` goals. Each such option is called a [`Candidate`]. If there are multiple applicable candidates, we prefer some candidates over others. We store the relevant information in their [`CandidateSource`].

This preference may result in incorrect inference or region constraints and would therefore be unsound during coherence. Because of this, we simply try to merge all candidates in coherence.

## `Trait` goals

Trait goals merge their applicable candidates in [`fn merge_trait_candidates`]. This document provides additional details and references to explain *why* we've got the current preference rules.

### `CandidateSource::BuiltinImpl(BuiltinImplSource::Trivial))`

Trivial builtin impls are builtin impls which are known to be always applicable for well-formed types. This means that if one exists, using another candidate should never have fewer constraints. We currently only consider `Sized` - and `MetaSized` - impls to be trivial.

This is necessary to prevent a lifetime error for the following pattern

```rust
trait Trait<T>: Sized {}
impl<'a> Trait<u32> for &'a str {}
impl<'a> Trait<i32> for &'a str {}
fn is_sized<T: Sized>(_: T) {}
fn foo<'a, 'b, T>(x: &'b str)
where
    &'a str: Trait<T>,
{
    // Elaborating the `&'a str: Trait<T>` where-bound results in a
    // `&'a str: Sized` where-bound. We do not want to prefer this
    // over the builtin impl. 
    is_sized(x);
}
```

This preference is incorrect in case the builtin impl has a nested goal which relies on a non-param where-clause
```rust
struct MyType<'a, T: ?Sized>(&'a (), T);
fn is_sized<T>() {}
fn foo<'a, T: ?Sized>()
where
    (MyType<'a, T>,): Sized,
    MyType<'static, T>: Sized,
{
    // The where-bound is trivial while the builtin `Sized` impl for tuples
    // requires proving `MyType<'a, T>: Sized` which can only be proven by
    // using the where-clause, adding an unnecessary `'static` constraint.
    is_sized::<(MyType<'a, T>,)>();
    //~^ ERROR lifetime may not live long enough
}
```

### `CandidateSource::ParamEnv`

Once there's at least one *non-global* `ParamEnv` candidate, we prefer *all* `ParamEnv` candidates over other candidate kinds.
A where-bound is global if it is not higher-ranked and doesn't contain any generic parameters. It may contain `'static`.

We try to apply where-bounds over other candidates as users tends to have the most control over them, so they can most easily
adjust them in case our candidate preference is incorrect.

#### Preference over `Impl` candidates

This is necessary to avoid region errors in the following example

```rust
trait Trait<'a> {}
impl<T> Trait<'static> for T {}
fn impls_trait<'a, T: Trait<'a>>() {}
fn foo<'a, T: Trait<'a>>() {
    impls_trait::<'a, T>();
}
```

We also need this as shadowed impls can result in currently ambiguous solver cycles: [trait-system-refactor-initiative#76]. Without preference we'd be forced to fail with ambiguity
errors if the where-bound results in region constraints to avoid incompleteness.
```rust
trait Super {
    type SuperAssoc;
}

trait Trait: Super<SuperAssoc = Self::TraitAssoc> {
    type TraitAssoc;
}

impl<T, U> Trait for T
where
    T: Super<SuperAssoc = U>,
{
    type TraitAssoc = U;
}

fn overflow<T: Trait>() {
    // We can use the elaborated `Super<SuperAssoc = Self::TraitAssoc>` where-bound
    // to prove the where-bound of the `T: Trait` implementation. This currently results in
    // overflow. 
    let x: <T as Trait>::TraitAssoc;
}
```

This preference causes a lot of issues. See [#24066]. Most of the
issues are caused by preferring where-bounds over impls even if the where-bound guides type inference:
```rust
trait Trait<T> {
    fn call_me(&self, x: T) {}
}
impl<T> Trait<u32> for T {}
impl<T> Trait<i32> for T {}
fn bug<T: Trait<U>, U>(x: T) {
    x.call_me(1u32);
    //~^ ERROR mismatched types
}
```
However, even if we only apply this preference if the where-bound doesn't guide inference, it may still result
in incorrect lifetime constraints:
```rust
trait Trait<'a> {}
impl<'a> Trait<'a> for &'a str {}
fn impls_trait<'a, T: Trait<'a>>(_: T) {}
fn foo<'a, 'b>(x: &'b str)
where
    &'a str: Trait<'b>
{
    // Need to prove `&'x str: Trait<'b>` with `'b: 'x`.
    impls_trait::<'b, _>(x);
    //~^ ERROR lifetime may not live long enough
}
```

#### Preference over `AliasBound` candidates

This is necessary to avoid region errors in the following example
```rust
trait Bound<'a> {}
trait Trait<'a> {
    type Assoc: Bound<'a>;
}

fn impls_bound<'b, T: Bound<'b>>() {}
fn foo<'a, 'b, 'c, T>()
where
    T: Trait<'a>,
    for<'hr> T::Assoc: Bound<'hr>,
{
    impls_bound::<'b, T::Assoc>();
    impls_bound::<'c, T::Assoc>();
}
```
It can also result in unnecessary constraints
```rust
trait Bound<'a> {}
trait Trait<'a> {
    type Assoc: Bound<'a>;
}

fn impls_bound<'b, T: Bound<'b>>() {}
fn foo<'a, 'b, T>()
where
    T: for<'hr> Trait<'hr>,
    <T as Trait<'b>>::Assoc: Bound<'a>,
{
    // Using the where-bound for `<T as Trait<'a>>::Assoc: Bound<'a>`
    // unnecessarily equates `<T as Trait<'a>>::Assoc` with the
    // `<T as Trait<'b>>::Assoc` from the env.
    impls_bound::<'a, <T as Trait<'a>>::Assoc>();
    // For a `<T as Trait<'b>>::Assoc: Bound<'b>` the self type of the
    // where-bound matches, but the arguments of the trait bound don't.
    impls_bound::<'b, <T as Trait<'b>>::Assoc>();
}
```

#### Why no preference for global where-bounds

Global where-bounds are either fully implied by an impl or unsatisfiable. If they are unsatisfiable, we don't really care what happens. If a where-bound is fully implied then using the impl to prove the trait goal cannot result in additional constraints. For trait goals this is only useful for where-bounds which use `'static`:

```rust
trait A {
    fn test(&self);
}

fn foo(x: &dyn A)
where
    dyn A + 'static: A, // Using this bound would lead to a lifetime error.
{
    x.test();
}
```
More importantly, by using impls here we prevent global where-bounds from shadowing impls when normalizing associated types. There are no known issues from preferring impls over global where-bounds.

#### Why still consider global where-bounds

Given that we just use impls even if there exists a global where-bounds, you may ask why we don't just ignore these global where-bounds entirely: we use them to weaken the inference guidance from non-global where-bounds.

Without a global where-bound, we currently prefer non-global where bounds even though there would be an applicable impl as well. By adding a non-global where-bound, this unnecessary inference guidance is disabled, allowing the following to compile:
```rust
fn check<Color>(color: Color)
where
    Vec: Into<Color> + Into<f32>,
{
    let _: f32 = Vec.into();
    // Without the global `Vec: Into<f32>`  bound we'd
    // eagerly use the non-global `Vec: Into<Color>` bound
    // here, causing this to fail.
}

struct Vec;
impl From<Vec> for f32 {
    fn from(_: Vec) -> Self {
        loop {}
    }
}
```

### `CandidateSource::AliasBound`

We prefer alias-bound candidates over impls. We currently use this preference to guide type inference, causing the following to compile. I personally don't think this preference is desirable 🤷
```rust
pub trait Dyn {
    type Word: Into<u64>;
    fn d_tag(&self) -> Self::Word;
    fn tag32(&self) -> Option<u32> {
        self.d_tag().into().try_into().ok()
        // prove `Self::Word: Into<?0>` and then select a method
        // on `?0`, needs eager inference.
    }
}
```
```rust
fn impl_trait() -> impl Into<u32> {
    0u16
}

fn main() {
    // There are two possible types for `x`:
    // - `u32` by using the "alias bound" of `impl Into<u32>`
    // - `impl Into<u32>`, i.e. `u16`, by using `impl<T> From<T> for T`
    //
    // We infer the type of `x` to be `u32` even though this is not
    // strictly necessary and can even lead to surprising errors.
    let x = impl_trait().into();
    println!("{}", std::mem::size_of_val(&x));
}
```
This preference also avoids ambiguity due to region constraints, I don't know whether people rely on this in practice.
```rust
trait Bound<'a> {}
impl<T> Bound<'static> for T {}
trait Trait<'a> {
    type Assoc: Bound<'a>;
}

fn impls_bound<'b, T: Bound<'b>>() {}
fn foo<'a, T: Trait<'a>>() {
    // Should we infer this to `'a` or `'static`.
    impls_bound::<'_, T::Assoc>();
}
```

### `CandidateSource::BuiltinImpl(BuiltinImplSource::Object(_))`

We prefer builtin trait object impls over user-written impls. This is **unsound** and should be remoed in the future. See [#57893](https://github.com/rust-lang/rust/issues/57893) and [#141347](https://github.com/rust-lang/rust/pull/141347) for more details.

## `NormalizesTo` goals

The candidate preference behavior during normalization is implemented in [`fn assemble_and_merge_candidates`].

### Where-bounds shadow impls

Normalization of associated items does not consider impls if the corresponding trait goal has been proven via a `ParamEnv` or `AliasBound` candidate.
This means that for where-bounds which do not constrain associated types, the associated types remain *rigid*.

This is necessary to avoid unnecessary region constraints from applying impls.
```rust
trait Trait<'a> {
    type Assoc;
}
impl Trait<'static> for u32 {
    type Assoc = u32;
}

fn bar<'b, T: Trait<'b>>() -> T::Assoc { todo!() }
fn foo<'a>()
where
    u32: Trait<'a>,
{
    // Normalizing the return type would use the impl, proving
    // the `T: Trait` where-bound would use the where-bound, resulting
    // in different region constraints.
    bar::<'_, u32>();
}
```

### We always consider `AliasBound` candidates

In case the where-bound does not specify the associated item, we consider `AliasBound` candidates instead of treating the alias as rigid, even though the trait goal was proven via a `ParamEnv` candidate.

```rust
trait Super {
    type Assoc;
}
trait Bound {
    type Assoc: Super<Assoc = u32>;
}
trait Trait: Super {}

// Elaborating the environment results in a `T::Assoc: Super` where-bound.
// This where-bound must not prevent normalization via the `Super<Assoc = u32>`
// item bound.
fn heck<T: Bound<Assoc: Trait>>(x: <T::Assoc as Super>::Assoc) -> u32 {
    x
}
```
Using such an alias can result in additional region constraints, cc [#133044].
```rust
trait Bound<'a> {
    type Assoc;
}
trait Trait {
    type Assoc: Bound<'static, Assoc = u32>;
}

fn heck<'a, T: Trait<Assoc: Bound<'a>>>(x: <T::Assoc as Bound<'a>>::Assoc) {
    // Normalizing the associated type requires `T::Assoc: Bound<'static>` as it
    // uses the `Bound<'static>` alias-bound instead of keeping the alias rigid.
    drop(x);
}
```

### We prefer `ParamEnv` candidates over `AliasBound`

While we use `AliasBound` candidates if the where-bound does not specify the associated type, in case it does, we prefer the where-bound.
This is necessary for the following example:
```rust
// Make sure we prefer the `I::IntoIterator: Iterator<Item = ()>`
// where-bound over the `I::Intoiterator: Iterator<Item = I::Item>`
// alias-bound.

trait Iterator {
    type Item;
}

trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
}

fn normalize<I: Iterator<Item = ()>>() {}

fn foo<I>()
where
    I: IntoIterator,
    I::IntoIter: Iterator<Item = ()>,
{
    // We need to prefer the `I::IntoIterator: Iterator<Item = ()>`
    // where-bound over the `I::Intoiterator: Iterator<Item = I::Item>`
    // alias-bound.
    normalize::<I::IntoIter>();
}
```

### We always consider where-bounds

Even if the trait goal was proven via an impl, we still prefer `ParamEnv` candidates, if any exist.

#### We prefer "orphaned" where-bounds

We add "orphaned" `Projection` clauses into the `ParamEnv` when normalizing item bounds of GATs and RPITIT in `fn check_type_bounds`.
We need to prefer these `ParamEnv` candidates over impls and other where-bounds. 
```rust
#![feature(associated_type_defaults)]
trait Foo {
    // We should be able to prove that `i32: Baz<Self>` because of
    // the impl below, which requires that `Self::Bar<()>: Eq<i32>`
    // which is true, because we assume `for<T> Self::Bar<T> = i32`.
    type Bar<T>: Baz<Self> = i32;
}
trait Baz<T: ?Sized> {}
impl<T: Foo + ?Sized> Baz<T> for i32 where T::Bar<()>: Eq<i32> {}
trait Eq<T> {}
impl<T> Eq<T> for T {}
```

I don't fully understand the cases where this preference is actually necessary and haven't been able to exploit this in fun ways yet, but 🤷

#### We prefer global where-bounds over impls

This is necessary for the following to compile. I don't know whether anything relies on it in practice 🤷
```rust
trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}

fn foo<T>(x: T) -> <u32 as Id>::This
where
    u32: Id<This = T>,
{
    x
}
```
This means normalization can result in additional region constraints, cc [#133044].
```rust
trait Trait {
    type Assoc;
}

impl Trait for &u32 {
    type Assoc = u32;
}

fn trait_bound<T: Trait>() {}
fn normalize<T: Trait<Assoc = u32>>() {}

fn foo<'a>()
where
    &'static u32: Trait<Assoc = u32>,
{
    trait_bound::<&'a u32>(); // ok, proven via impl
    normalize::<&'a u32>(); // error, proven via where-bound
}
```

[`Candidate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_next_trait_solver/solve/assembly/struct.Candidate.html
[`CandidateSource`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_next_trait_solver/solve/enum.CandidateSource.html
[`fn merge_trait_candidates`]: https://github.com/rust-lang/rust/blob/e3ee7f7aea5b45af3b42b5e4713da43876a65ac9/compiler/rustc_next_trait_solver/src/solve/trait_goals.rs#L1342-L1424
[`fn assemble_and_merge_candidates`]: https://github.com/rust-lang/rust/blob/e3ee7f7aea5b45af3b42b5e4713da43876a65ac9/compiler/rustc_next_trait_solver/src/solve/assembly/mod.rs#L920-L1003
[trait-system-refactor-initiative#76]: https://github.com/rust-lang/trait-system-refactor-initiative/issues/76
[#24066]: https://github.com/rust-lang/rust/issues/24066
[#133044]: https://github.com/rust-lang/rust/issues/133044