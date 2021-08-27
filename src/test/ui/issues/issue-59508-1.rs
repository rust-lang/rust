#![allow(dead_code)]
#![feature(const_generics_defaults)]

// This test checks that generic parameter re-ordering diagnostic suggestions mention that
// consts come after types and lifetimes when the `const_generics_defaults` feature is enabled.
// We cannot run rustfix on this test because of the above const generics warning.

struct A;

impl A {
    pub fn do_things<T, 'a, 'b: 'a>() {
    //~^ ERROR lifetime parameters must be declared prior to type parameters
        println!("panic");
    }
}

fn main() {}
