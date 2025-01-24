//@ run-pass
#![allow(dead_code)]
// Test that the compiler considers the 'static bound declared in the
// trait. Issue #20890.


trait Foo {
    type Value: 'static;
    fn dummy(&self) { }
}

fn require_static<T: 'static>() {}

fn takes_foo<F: Foo>() {
    require_static::<F::Value>()
}

fn main() { }
