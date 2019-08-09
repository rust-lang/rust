#![allow(dead_code)]
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

// This test checks that generic parameter re-ordering diagnostic suggestions mention that
// consts come after types and lifetimes when the `const_generics` feature is enabled.
// We cannot run rustfix on this test because of the above const generics warning.

struct A;

impl A {
    pub fn do_things<T, 'a, 'b: 'a>() {
    //~^ ERROR lifetime parameters must be declared prior to type parameters
        println!("panic");
    }
}

fn main() {}
