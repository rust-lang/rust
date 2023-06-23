#![crate_name="foo"]

// @files foo '["index.html", "all.html", "sidebar-items.js"]'
// @!has "foo/struct.Foo.html"
#[doc(hidden)]
pub struct Foo;

// @!has "foo/struct.Bar.html"
pub use crate::Foo as Bar;

// @!has "foo/struct.Baz.html"
#[doc(hidden)]
pub use crate::Foo as Baz;

// @!has "foo/foo/index.html"
#[doc(hidden)]
pub mod foo {}

// @!has "foo/bar/index.html"
pub use crate::foo as bar;

// @!has "foo/baz/index.html"
#[doc(hidden)]
pub use crate::foo as baz;
