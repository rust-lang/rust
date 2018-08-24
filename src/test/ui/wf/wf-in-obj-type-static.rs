// Check that we enforce WF conditions also for types in fns.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Object<T> { }

struct MustBeCopy<T:Copy> {
    t: T
}

struct Foo<T> {
    // needs T: 'static
    x: Object<&'static T> //~ ERROR E0310
}

#[rustc_error]
fn main() { }
