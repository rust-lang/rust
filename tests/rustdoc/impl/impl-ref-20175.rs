// https://github.com/rust-lang/rust/issues/20175

#![crate_name="issue_20175"]

pub trait Foo {
    fn foo(&self) {}
}

pub struct Bar;

//@ has issue_20175/struct.Bar.html \
//      '//*[@id="method.foo"]' \
//      'fn foo'
impl<'a> Foo for &'a Bar {}
