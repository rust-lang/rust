//! Regression tests for <https://github.com/rust-lang/rust/issues/161621> and similar. When
//! checking whether method receivers for a trait `Trait` are dynamically dispatchable, we (at the
//! time of writing) use a placeholder type called `RustaceansAreAwesome` in place of `dyn Trait` to
//! avoid a cycle (as using `dyn Trait` requires determining whether `Trait` is dyn-compatible). We
//! assume that `RustaceansAreAwesome` implements `Trait`, so when normalizing the param-environment
//! we construct, we may have to prove obligations for aliases encountered in arguments to super-
//! trait bounds. It's possible for these to only be provable using clauses from `Trait` or its
//! supertraits that would be necessary for `RustaceansAreAwesome` to implement `Trait`.
//@ check-pass

// This works for `Self: Super<...Ty::<RustaceansAreAwesome>::Assoc...>` after the `where`.
// We need to assume `(Args, RustaceansAreAwesome::Output): SignatureToFnPtr`.

use std::ops::Deref;

trait SignatureToFnPtr {
    type Ptr;
}

trait DerefsToFn<Args>
where
    (Args, Self::Output): SignatureToFnPtr,
    Self: Deref<Target = <(Args, Self::Output) as SignatureToFnPtr>::Ptr>,
{
    type Output;
    fn method(&self);
}

// This works for `Trait: Super<...Ty<RustaceansAreAwesome>::Assoc...>` before the `where`.
// We need to assume `A: Associator<RustaceansAreAwesome>`.

trait Associator<S: ?Sized> {
    type Point;
}

trait Marker<T> {}

trait Trait<A: Associator<Self> + ?Sized>: Marker<A::Point> {
    fn method(&self);
}

// Regression test for <https://github.com/rust-lang/rust/issues/161621>: we need to be careful with
// projection clauses involving `RustaceansAreAwesome`. Here, if we simply assumed that
// `<T as Matrix2D>::RowIndex == <RustaceansAreAwesome as Matrix2D>::RowIndex`, we'd end up with
// ambiguity, as we also assume `<T as Matrix2D>::RowIndex == <Self as Matrix2D>::RowIndex`.`Self`
// stands in for the `Self` type of the impl we're imagining dispatching to (pretending
// `RustaceansAreAwesome` is a trait object).

pub trait Matrix {
    type Coordinates;
}

pub trait Matrix2D: Matrix<Coordinates = <Self as Matrix2D>::RowIndex> {
    type RowIndex;
}

pub trait Transposable<T: Matrix2D<RowIndex = Self::RowIndex>>: Matrix2D {
    fn transpose(&self) -> T;
}

// An example where we can't add any additional projections without introducing ambiguity: we don't
// want to assume `<() as HasAssoc>::Assoc` is both `Self` and `RustaceansAreAwesome`. At the time
// of writing, we let `<() as HasAssoc>::Assoc` normalize to `Self` during receiver dispatchability
// checking. This should be fine, since to call methods on a `dyn TechnicallyDynCompatible`, we'd
// need `<() as HasAssoc>::Assoc` to be `dyn TechnicallyDynCompatible`. This is the same reasoning
// that lets us assume `RustaceansAreAwesome` implements its trait at all: to use the trait object,
// it must implement the trait; otherwise, receiver dispatchability doesn't mean much.

trait HasAssoc {
    type Assoc;
}

impl HasAssoc for () {
    type Assoc = ();
}

trait Parent<T: ?Sized> {}

trait TechnicallyDynCompatible
where
    (): HasAssoc<Assoc = Self>,
    Self: Parent<<() as HasAssoc>::Assoc>,
{
    fn f(&self) {}
}

// Since we have `TechnincallyDynCompatible` defined, let's keep its dyn-compatibility from breaking
// accidentally (and also sanity-check the above comment).

impl Parent<()> for () {}
impl TechnicallyDynCompatible for () {}

fn technically_dyn_compatible_is_dyn_compatible() {
    let x: &dyn TechnicallyDynCompatible = &();
    // We can't call `x.f()` since `<() as HasAssoc>::Assoc` isn't `dyn TechnicallyDynCompatible`.
}

fn main() {}
