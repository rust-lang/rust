// check-pass
// aux-build: inner-crate-enum.rs
// compile-flags:-Z unstable-options --output-format json

#[doc(inline)]
pub extern crate inner_crate_enum;

fn main() {}
