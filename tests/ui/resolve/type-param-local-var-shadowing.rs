//! Test that items in subscopes correctly shadow type parameters and local variables
//!
//! Regression test for https://github.com/rust-lang/rust/issues/23880

//@ run-pass

#![allow(unused)]
struct Foo<X> {
    x: Box<X>,
}
impl<Bar> Foo<Bar> {
    fn foo(&self) {
        type Bar = i32;
        let _: Bar = 42;
    }
}

fn main() {
    let f = 1;
    {
        fn f() {}
        f();
    }
}
