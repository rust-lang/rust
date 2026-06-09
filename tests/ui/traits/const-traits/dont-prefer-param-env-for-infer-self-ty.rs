//@ check-pass

#![feature(const_trait_impl)]

const trait Foo {}

impl<T> const Foo for (T,) where T: [const] Foo {}

const fn needs_const_foo(_: impl [const] Foo + Copy) {}

const fn test<T: [const] Foo + Copy>(t: T) {
    needs_const_foo((t,));
}

fn main() {}
