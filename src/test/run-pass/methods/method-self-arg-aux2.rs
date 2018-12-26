// run-pass
// Test method calls with self as an argument (cross-crate)

#![feature(box_syntax)]

// aux-build:method_self_arg2.rs
extern crate method_self_arg2;
use method_self_arg2::{Foo, Bar};

fn main() {
    let x = Foo;
    // Test external call.
    Bar::foo1(&x);
    Bar::foo2(x);
    Bar::foo3(box x);

    Bar::bar1(&x);
    Bar::bar2(x);
    Bar::bar3(box x);

    x.run_trait();

    assert_eq!(method_self_arg2::get_count(), 2*2*3*3*5*5*7*7*11*11*13*13*17);
}
