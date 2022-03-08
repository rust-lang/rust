#![crate_name = "foo"]

pub trait Trait {
    const FOO: u32 = 12;

    fn foo();
}

pub struct Bar;

// @has 'foo/struct.Bar.html'
// @has - '//h3[@class="sidebar-title"]' 'Associated Constants'
// @has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Trait for Bar {
    const FOO: u32 = 1;

    fn foo() {}
}

pub enum Foo {
    A,
}

// @has 'foo/enum.Foo.html'
// @has - '//h3[@class="sidebar-title"]' 'Associated Constants'
// @has - '//div[@class="sidebar-elems"]//a' 'FOO'
impl Trait for Foo {
    const FOO: u32 = 1;

    fn foo() {}
}
