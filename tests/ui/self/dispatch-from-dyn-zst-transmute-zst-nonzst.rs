// We used to allow erroneous `DispatchFromDyn` impls whose RHS type contained
// fields that weren't ZSTs. I don't believe this was possible to abuse, but
// it's at least nice to give users better errors.

#![feature(arbitrary_self_types)]
#![feature(unsize)]
#![feature(dispatch_from_dyn)]

use std::marker::Unsize;
use std::ops::DispatchFromDyn;

struct Dispatchable<T: ?Sized, Z> {
    _ptr: Box<T>,
    z: Z,
}

impl<T, U> DispatchFromDyn<Dispatchable<U, i32>> for Dispatchable<T, ()>
//~^ ERROR implementing `DispatchFromDyn` does not allow multiple fields to be coerced
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

fn main() {}
