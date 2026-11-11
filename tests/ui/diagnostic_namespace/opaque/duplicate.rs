#![crate_type = "lib"]
#![feature(diagnostic_opaque)]
#![deny(unused_attributes)]
#[diagnostic::opaque]
#[diagnostic::opaque]
//~^ERROR unused attribute
macro_rules! m {
    () => {}
}
