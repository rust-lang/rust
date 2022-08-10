#![feature(type_alias_impl_trait)]

trait MyTrait {}
impl MyTrait for i32 {}

// @hastext impl_trait_alias/type.Foo.html 'Foo'
/// debug type
pub type Foo = impl MyTrait;

// @hastext impl_trait_alias/fn.foo.html 'foo'
/// debug function
pub fn foo() -> Foo {
    1
}
