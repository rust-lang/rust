// https://github.com/rust-lang/rust/issues/33592
#![crate_name = "foo"]

pub trait Foo<T> {}

pub struct Bar;

pub struct Baz;

//@ has foo/trait.Foo.html '//h3[@class="code-header"]' 'impl Foo<i32> for Bar'
impl Foo<i32> for Bar {}

//@ has foo/trait.Foo.html '//h3[@class="code-header"]' 'impl<T> Foo<T> for Baz'
impl<T> Foo<T> for Baz {}
