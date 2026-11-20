//@ check-pass

#![feature(impl_trait_in_bindings)]

trait Foo {}
impl Foo for () {}

fn main() {
    let x: impl Foo = ();
}
