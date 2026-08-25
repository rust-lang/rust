//@ run-rustfix
#![allow(dead_code)]
mod first {
    trait Foo { fn m(self: Box<Self>); }
    fn foo<T: Foo>(a: T) {
        a.m(); //~ ERROR no method named `m` found
    }
}
mod second {
    use std::sync::Arc;
    trait Bar { fn m(self: Arc<Self>); }
    fn bar(b: impl Bar) {
        b.m(); //~ ERROR no method named `m` found
    }
}

fn main() {}
