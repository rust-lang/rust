//! Regression test for <https://github.com/rust-lang/rust/issues/153370>.

//@ edition:2024
//@ compile-flags: -Znext-solver=globally

#![feature(transmutability)]

trait NodeImpl {}
struct Wrap<F, P>(F, P);
impl<F, P> Wrap<F, P> {
    fn new(_: F) -> Self {
        loop {}
    }
}

impl<F, A> NodeImpl for Wrap<F, (A,)> where F: std::mem::TransmuteFrom<()> {}
fn trigger_ice() {
    let _: &dyn NodeImpl = &Wrap::<_, (i128,)>::new(async |_: &(), i128| 0);
    //~^ ERROR type annotations needed
}

fn main() {}
