//! Checks for invalid uses of the `cfi_encoding` attribute

#![feature(cfi_encoding)]
#![crate_type = "lib"]

#[cfi_encoding] //~ ERROR malformed `cfi_encoding` attribute input
pub struct Type1(i32);

#[cfi_encoding = "Foo"] //~ ERROR  the `cfi_encoding` attribute cannot be used on traits
pub trait X {}

#[cfi_encoding = "Bar"] //~ ERROR  the `cfi_encoding` attribute cannot be used on type aliases
pub type Y = Type1;
