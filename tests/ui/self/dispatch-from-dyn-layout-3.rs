//@ check-pass

// Make sure that object safety checking doesn't freak out when
// we have impossible-to-satisfy `DispatchFromDyn` predicates.

#![feature(dispatch_from_dyn)]
#![feature(arbitrary_self_types)]

use std::ops::Deref;
use std::ops::DispatchFromDyn;

trait Trait<T: Deref<Target = Self>>
where
    for<'a> &'a T: DispatchFromDyn<&'a T>,
{
    fn foo(self: &T) -> Box<dyn Trait<T>>;
}

fn main() {}
