//@ known-bug: #119694
#![feature(dyn_star)]

trait Trait {
    fn foo(self);
}

impl Trait for usize {
    fn foo(self) {}
}

fn bar(x: dyn* Trait) {
    x.foo();
}

fn main() {
    bar(0usize);
}
