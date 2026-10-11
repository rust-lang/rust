#![crate_name="foo"]

//@ files foo '["index.html", "all.html", "sidebar-items.js"]'
//@ !has "foo/type.Foo.html"
#[doc(hidden)]
pub struct Foo;

//@ !has "foo/type.Bar.html"
pub use crate::Foo as Bar;

//@ !has "foo/type.Baz.html"
#[doc(hidden)]
pub use crate::Foo as Baz;

//@ !has "foo/foo/index.html"
#[doc(hidden)]
pub mod foo {}

//@ !has "foo/bar/index.html"
pub use crate::foo as bar;

//@ !has "foo/baz/index.html"
#[doc(hidden)]
pub use crate::foo as baz;
