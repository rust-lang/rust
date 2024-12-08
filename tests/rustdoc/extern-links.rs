//@ aux-build:extern-links.rs
//@ ignore-cross-compile

#![crate_name = "foo"]

pub extern crate extern_links;

//@ !has foo/index.html '//a' 'extern_links'
#[doc(no_inline)]
pub use extern_links as extern_links2;

//@ !has foo/index.html '//a' 'Foo'
#[doc(no_inline)]
pub use extern_links::Foo;

#[doc(hidden)]
pub mod hidden {
    //@ !has foo/hidden/extern_links/index.html
    //@ !has foo/hidden/extern_links/struct.Foo.html
    pub use extern_links;
}
