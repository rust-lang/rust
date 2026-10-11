#![crate_name = "foo"]

pub trait Trait {
    const FOO: u32 = 12;

    fn foo();
}

pub struct Bar;

//@ has 'foo/type.Bar.html'
//@ !has - '//div[@class="sidebar-elems"]//h3' 'Associated Constants'
//@ !has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Trait for Bar {
    const FOO: u32 = 1;

    fn foo() {}
}

pub enum Foo {
    A,
}

//@ has 'foo/type.Foo.html'
//@ !has - '//div[@class="sidebar-elems"]//h3' 'Associated Constants'
//@ !has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Trait for Foo {
    const FOO: u32 = 1;

    fn foo() {}
}

pub struct Baz;

//@ has 'foo/type.Baz.html'
//@ has - '//div[@class="sidebar-elems"]//h3' 'Associated Constants'
//@ has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Baz {
    pub const FOO: u32 = 42;
}

pub enum Quux {
    B,
}

//@ has 'foo/type.Quux.html'
//@ has - '//div[@class="sidebar-elems"]//h3' 'Associated Constants'
//@ has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Quux {
    pub const FOO: u32 = 42;
}
