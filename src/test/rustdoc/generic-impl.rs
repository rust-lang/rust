#![crate_name = "foo"]

use std::fmt;

// @!has foo/struct.Bar.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
pub struct Bar;

// @has foo/struct.Foo.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
pub struct Foo;
// @has foo/struct.Foo.html '//div[@class="sidebar-links"]/a[@href="#impl-ToString"]' 'ToString'

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo")
    }
}
