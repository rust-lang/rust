//@ known-bug: #120482
#![feature(object_safe_for_dispatch)]

trait B {
    fn bar(&self, x: &Self);
}

trait A {
    fn g(new: B) -> B;
}

fn main() {}
