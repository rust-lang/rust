#![feature(type_alias_impl_trait)]

type Foo = (impl Sized, u8);
pub fn foo() -> Foo {
    //~^ ERROR private type alias `Foo` in public interface
    (42, 42)
}
fn main() {}
