//@ run-pass
//@ aux-build:trait_safety_lib.rs

// Simple smoke test that unsafe traits can be compiled across crates.


extern crate trait_safety_lib as lib;

use lib::Foo;

struct Bar { x: isize }
unsafe impl Foo for Bar {
    fn foo(&self) -> isize { self.x }
}

fn take_foo<F:Foo>(f: &F) -> isize { f.foo() }

fn main() {
    let x: isize = 22;
    assert_eq!(22, take_foo(&x));

    let x: Bar = Bar { x: 23 };
    assert_eq!(23, take_foo(&x));
}
