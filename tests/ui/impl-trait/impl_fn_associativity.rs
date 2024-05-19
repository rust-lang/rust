#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn f_debug() -> impl Fn() -> impl Debug {
    //~^ ERROR undefined opaque type
    || ()
}

fn ff_debug() -> impl Fn() -> impl Fn() -> impl Debug {
    //~^ ERROR undefined opaque type
    //~| ERROR undefined opaque type
    || f_debug()
}

fn multi() -> impl Fn() -> (impl Debug + Send) {
    //~^ ERROR undefined opaque type
    || ()
}

fn main() {
    // Check that `ff_debug` is `() -> (() -> Debug)` and not `(() -> ()) -> Debug`
    let debug = ff_debug()()();
    assert_eq!(format!("{:?}", debug), "()");

    let x = multi()();
    assert_eq!(format!("{:?}", x), "()");
    fn assert_send(_: &impl Send) {}
    assert_send(&x);
}
