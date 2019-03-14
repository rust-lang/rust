#![allow(plugin_as_library)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// aux-build:macro-crate-test.rs
// ignore-stage1

#![feature(rustc_attrs)]

#[macro_use]
extern crate macro_crate_test;

// The duplicate macro will create a copy of the item with the given identifier.

#[rustc_duplicate(MyCopy)]
struct MyStruct {
    number: i32
}

trait TestTrait {
    #[rustc_duplicate(TestType2)]
    type TestType;

    #[rustc_duplicate(required_fn2)]
    fn required_fn(&self);

    #[rustc_duplicate(provided_fn2)]
    fn provided_fn(&self) { }
}

impl TestTrait for MyStruct {
    #[rustc_duplicate(TestType2)]
    type TestType = f64;

    #[rustc_duplicate(required_fn2)]
    fn required_fn(&self) { }
}

fn main() {
    let s = MyStruct { number: 42 };
    s.required_fn();
    s.required_fn2();
    s.provided_fn();
    s.provided_fn2();

    let s = MyCopy { number: 42 };
}
