//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+Inline,+GVN -Zvalidate-mir

#![feature(unsize)]

use std::marker::Unsize;

pub trait CastTo<U: ?Sized>: Unsize<U> {}

// Not well-formed!
impl<T: ?Sized, U: ?Sized> CastTo<U> for T {}
//~^ ERROR the trait bound `T: Unsize<U>` is not satisfied

pub trait Cast {
    fn cast<U: ?Sized>(&self)
    where
        Self: CastTo<U>;
}
impl<T: ?Sized> Cast for T {
    #[inline(always)]
    fn cast<U: ?Sized>(&self)
    where
        Self: CastTo<U>,
    {
        let x: &U = self;
    }
}

fn main() {
    // When we inline this call, then we run GVN, then
    // GVN tries to evaluate the `() -> [i32]` unsize.
    // That's invalid!
    ().cast::<[i32]>();
}
