//@ compile-flags: -Z unstable-options --generate-redirect-map

#![crate_name = "foo"]

//@ !has foo/private/struct.Quz.html
//@ !has foo/hidden/struct.Bar.html
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
