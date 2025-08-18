#![cfg_attr(feature_enabled, feature(const_trait_impl))]

trait Foo {
    fn a(&self);
}

trait Bar: [const] Foo {}

const fn foo<T: [const] Bar>(x: &T) {
    x.a();
}

fn main() {}
