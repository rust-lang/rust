#![crate_name = "foo"]

use std::fmt;

//@ !has foo/type.Bar.html '//*[@id="impl-ToString-for-Bar"]' ''
pub struct Bar;

//@ has foo/type.Foo.html '//*[@id="impl-ToString-for-T"]//h3[@class="code-header"]' 'impl<T> ToString for T'
pub struct Foo;
//@ has foo/type.Foo.html '//*[@class="sidebar-elems"]//section//a[@href="#impl-ToString-for-T"]' 'ToString'

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo")
    }
}
