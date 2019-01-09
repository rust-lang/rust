// aux-build:attribute-with-error.rs

#![feature(custom_inner_attributes)]

extern crate attribute_with_error;

use attribute_with_error::foo;

#[foo]
fn test1() {
    let a: i32 = "foo";
    //~^ ERROR: mismatched types
    let b: i32 = "f'oo";
    //~^ ERROR: mismatched types
}

fn test2() {
    #![foo]

    // FIXME: should have a type error here and assert it works but it doesn't
}

trait A {
    // FIXME: should have a #[foo] attribute here and assert that it works
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

struct B;

impl A for B {
    #[foo]
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

#[foo]
fn main() {
}
