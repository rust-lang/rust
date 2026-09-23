//! Regression tests for <https://github.com/rust-lang/rust/issues/161621>. When checking whether
//! method receivers for a trait `Trait` are dynamically dispatchable, we use a placeholder type in
//! place of `dyn Trait` to avoid a cycle (as that would require determining whether `Trait` is dyn-
//! compatible). This placeholder must satisfy `Trait`'s where-bounds to avoid errors in param-
//! environment normalization.
//@ check-pass

// Test 1

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

// Test 2

trait Associator<S: ?Sized> {
    type Point;
}

trait Marker<T> {}

trait Trait<A: Associator<Self> + ?Sized>: Marker<A::Point> {
    fn method(&self);
}

fn main() {}
