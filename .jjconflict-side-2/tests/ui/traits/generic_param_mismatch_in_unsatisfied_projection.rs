//! This test used to ICE: #121134
//! The issue is that we're trying to prove a projection, but there's
//! no bound for the projection's trait, and the projection has the wrong
//! kind of generic parameter (lifetime vs type).
//! When actually calling the function with those broken bounds, trying to
//! instantiate the bounds with inference vars would ICE.
#![feature(unboxed_closures)]

trait Output<'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(&mut self, _: F)
    where
        F: for<'a> FnOnce(<F as Output<i32>>::Type),
        //~^ ERROR: trait takes 0 generic arguments but 1 generic argument was supplied
    {
    }
}

fn main() {
    let mut wrapper = Wrapper;
    wrapper.do_something_wrapper(|value| ());
}
