//@ compile-flags: -Z unstable-options --generate-redirect-map

#![crate_name = "foo"]

//@ !has foo/private/type.Quz.html
//@ !has foo/hidden/type.Bar.html
//@ has foo/redirect-map.json
pub use private::Quz;
pub use hidden::Bar;

mod private {
    pub struct Quz;
}

#[doc(hidden)]
pub mod hidden {
    pub struct Bar;
}

#[macro_export]
macro_rules! foo {
  () => {}
}
