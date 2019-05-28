// aux-build:test-macros.rs

#![feature(custom_inner_attributes)]

#[macro_use]
extern crate test_macros;

#[recollect_attr]
fn test1() {
    let a: i32 = "foo";
    //~^ ERROR: mismatched types
    let b: i32 = "f'oo";
    //~^ ERROR: mismatched types
}

fn test2() {
    #![recollect_attr]

    // FIXME: should have a type error here and assert it works but it doesn't
}

trait A {
    // FIXME: should have a #[recollect_attr] attribute here and assert that it works
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

struct B;

impl A for B {
    #[recollect_attr]
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

#[recollect_attr]
fn main() {
}
