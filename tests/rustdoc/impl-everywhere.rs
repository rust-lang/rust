#![crate_name = "foo"]

pub trait Foo {}
pub trait Foo2 {}

pub struct Bar;

impl Foo for Bar {}
impl Foo2 for Bar {}

//@ has foo/fn.foo.html '//section[@id="main-content"]//pre' "x: &'x impl Foo"
//@ has foo/fn.foo.html '//section[@id="main-content"]//pre' "-> &'x impl Foo"
pub fn foo<'x>(x: &'x impl Foo) -> &'x impl Foo {
    x
}

//@ has foo/fn.foo2.html '//section[@id="main-content"]//pre' "x: &'x impl Foo"
//@ has foo/fn.foo2.html '//section[@id="main-content"]//pre' '-> impl Foo2'
pub fn foo2<'x>(_x: &'x impl Foo) -> impl Foo2 {
    Bar
}

//@ has foo/fn.foo_foo.html '//section[@id="main-content"]//pre' '-> impl Foo + Foo2'
pub fn foo_foo() -> impl Foo + Foo2 {
    Bar
}

//@ has foo/fn.foo_foo_foo.html '//section[@id="main-content"]//pre' "x: &'x (impl Foo + Foo2)"
pub fn foo_foo_foo<'x>(_x: &'x (impl Foo + Foo2)) {
}
