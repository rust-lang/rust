#![crate_name="foo"]

//@ files "foo" \
// '["index.html", "all.html", "sidebar-items.js", "foo", "bar", "private", "struct.Bar.html", \
//   "type.Bar.html"]'
//@ files "foo/bar" '["index.html", "sidebar-items.js"]'

//@ !has "foo/priv/index.html"
//@ !has "foo/priv/type.Foo.html"
mod private {
    pub struct Foo;
}

//@ has 'foo/struct.Bar.html'
//@ matchesraw - '<meta http-equiv="refresh" content="0;URL=type.Bar.html">'
//@ has "foo/type.Bar.html"
pub use crate::private::Foo as Bar;

//@ !has "foo/foo/index.html"
mod foo {
    pub mod subfoo {}
}

//@ has "foo/bar/index.html"
pub use crate::foo::subfoo as bar;
