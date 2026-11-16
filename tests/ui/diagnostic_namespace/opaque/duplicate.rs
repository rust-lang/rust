#![crate_type = "lib"]
#![feature(diagnostic_opaque)]
#![deny(unused_attributes)]
#[diagnostic::opaque]
#[diagnostic::opaque]
//~^ERROR duplicate `diagnostic::opaque` attributes
macro_rules! m {
    () => {};
}
