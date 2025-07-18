// Tests a few different invalid repr attributes

// This is a regression test for https://github.com/rust-lang/rust/issues/143522
#![repr]
//~^ ERROR malformed `repr` attribute input [E0539]
//~| ERROR `repr` attribute cannot be used at crate level

// This is a regression test for https://github.com/rust-lang/rust/issues/143479
#[repr(align(0))]
//~^ ERROR invalid `repr(align)` attribute: not a power of two
//~| ERROR unsupported representation for zero-variant enum [E0084]
enum Foo {}

fn main() {}
