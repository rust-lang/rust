#![crate_name="foo"]

//@ files "foo" \
// '["index.html", "all.html", "sidebar-items.js", "foo", "bar", "private", "struct.Bar.html"]'
//@ files "foo/bar" '["index.html", "sidebar-items.js"]'

//@ !has "foo/priv/index.html"
//@ !has "foo/priv/struct.Foo.html"
mod private {
    pub struct Foo;
}

//@ has "foo/struct.Bar.html"
pub use crate::private::Foo as Bar;

//@ !has "foo/foo/index.html"
mod foo {
    pub mod subfoo {}
}

//@ has "foo/bar/index.html"
pub use crate::foo::subfoo as bar;
