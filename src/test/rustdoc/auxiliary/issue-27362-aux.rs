// compile-flags: -Cmetadata=aux

pub const fn foo() {}
pub const unsafe fn bar() {}

pub struct Foo;

impl Foo {
    pub const unsafe fn baz() {}
}
