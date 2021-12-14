// check-fail
// See issue #91899. If we treat unnormalized args as WF, `Self` can also be a
// source of unsoundness.

pub trait Yokeable<'a>: 'static {
    type Output: 'a;
}

impl<'a, T: 'static + ?Sized> Yokeable<'a> for &'static T {
    type Output = &'a T;
}

pub trait ZeroCopyFrom<C: ?Sized>: for<'a> Yokeable<'a> {
    /// Clone the cart `C` into a [`Yokeable`] struct, which may retain references into `C`.
    fn zero_copy_from<'b>(cart: &'b C) -> <Self as Yokeable<'b>>::Output;
}

impl<T> ZeroCopyFrom<[T]> for &'static [T] {
    fn zero_copy_from<'b>(cart: &'b [T]) -> &'b [T] {
        //~^ the parameter
        cart
    }
}

fn main() {}
