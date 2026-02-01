//@ compile-flags: -Znext-solver=globally
#![feature(type_alias_impl_trait)]

type Foo = Vec<impl Send>;

#[define_opaque(Foo)]
fn make_foo() -> Foo {}
//~^ ERROR type mismatch resolving

fn main() {}
