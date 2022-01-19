// run-pass
use std::fmt::Debug;

fn f_debug() -> impl Fn() -> impl Debug {
    || ()
}

fn ff_debug() -> impl Fn() -> impl Fn() -> impl Debug {
    || f_debug()
}

fn main() {
    // Check that `ff_debug` is `() -> (() -> Debug)` and not `(() -> ()) -> Debug`
    let debug = ff_debug()()();
    assert_eq!(format!("{:?}", debug), "()");
}
