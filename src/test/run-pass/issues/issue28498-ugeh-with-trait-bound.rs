// run-pass
#![allow(deprecated)] // FIXME: switch to `#[may_dangle]` below.

// Demonstrate the use of the unguarded escape hatch with a trait bound
// to assert that destructor will not access any dead data.
//
// Compare with compile-fail/issue28498-reject-trait-bound.rs

#![feature(dropck_parametricity)]

use std::fmt;

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<T:fmt::Debug>(u32, T);

impl<T:fmt::Debug> Drop for Foo<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        // Use of `unsafe_destructor_blind_to_params` is sound,
        // because destructor never accesses the `Debug::fmt` method
        // of `T`, despite having it available.
        println!("Dropping Foo({}, _)", self.0);
    }
}

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped);
    foo1 = Foo(1, &first_dropped);

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
