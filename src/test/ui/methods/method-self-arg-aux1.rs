// run-pass
// Test method calls with self as an argument (cross-crate)

#![feature(box_syntax)]

// aux-build:method_self_arg1.rs
extern crate method_self_arg1;
use method_self_arg1::Foo;

fn main() {
    let x = Foo;
    // Test external call.
    Foo::bar(&x);
    Foo::baz(x);
    Foo::qux(box x);

    x.foo(&x);

    assert_eq!(method_self_arg1::get_count(), 2*3*3*3*5*5*5*7*7*7);
}
