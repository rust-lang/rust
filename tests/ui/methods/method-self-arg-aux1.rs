//@ run-pass
// Test method calls with self as an argument (cross-crate)

//@ aux-build:method_self_arg1.rs
extern crate method_self_arg1;
use method_self_arg1::Foo;

fn main() {
    let x = Foo;
    // Test external call.
    Foo::bar(&x);
    Foo::baz(x);
    Foo::qux(Box::new(x));

    x.foo(&x);

    assert_eq!(method_self_arg1::get_count(), 2*3*3*3*5*5*5*7*7*7);
}
