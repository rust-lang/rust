//@ check-pass

#![feature(type_alias_impl_trait)]

pub type Foo = impl Fn() -> usize;
#[define_opaque(Foo)]
pub const fn bar() -> Foo {
    || 0usize
}
const BAZR: Foo = bar();

fn main() {}
