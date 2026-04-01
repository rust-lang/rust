//@ check-pass

#![feature(type_alias_impl_trait)]

type Foo = (impl Sized, u8);
#[define_opaque(Foo)]
pub fn foo() -> Foo {
    //~^ WARNING type alias `Foo` is more private than the item `foo`
    (42, 42)
}
fn main() {}
