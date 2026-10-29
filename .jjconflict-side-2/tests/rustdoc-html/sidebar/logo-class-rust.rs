#![feature(rustdoc_internals)]
#![allow(internal_features)]
#![doc(rust_logo)]
// Note: this test is paired with logo-class.rs and logo-class-default.rs.
//@ has logo_class_rust/struct.SomeStruct.html '//*[@class="logo-container"]/img[@class="rust-logo"]' ''
pub struct SomeStruct;
