//@ check-pass

// Make sure we don't prefer a subtrait that we would've otherwise eliminated
// in `consider_probe` during method probing.

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

struct W<T>(T);

trait Upstream {
    fn hello(&self) {}
}
impl<T> Upstream for T {}

trait Downstream: Upstream {
    fn hello(&self) {}
}
impl<T> Downstream for W<T> where T: Foo {}

trait Foo {}

fn main() {
    let x = W(1i32);
    x.hello();
}
