// Makes sure we don't ICE when encountering a receiver that is *ostensibly* dyn safe,
// because it satisfies `&dyn Bar: DispatchFromDyn<&dyn Bar>`, but is not a valid receiver
// in wfcheck.

#![feature(arbitrary_self_types)]

use std::ops::Deref;

trait Foo: Deref<Target = dyn Bar> {
     fn method(self: &dyn Bar) {}
     //~^ ERROR invalid `self` parameter type: `&dyn Bar`
}

trait Bar {}

fn test(x: &dyn Foo) {
     x.method();
}

fn main() {}
