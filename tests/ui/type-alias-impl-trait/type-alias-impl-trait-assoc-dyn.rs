//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Foo = Box<dyn Iterator<Item = impl Send>>;

#[define_opaque(Foo)]
fn make_foo() -> Foo {
    Box::new(vec![1, 2, 3].into_iter())
}

fn main() {}
