// compile-pass
// edition:2018

// Macro imported with `#[macro_use] extern crate`
use vec as imported_vec;

// Built-in attribute
use inline as imported_inline;

// Tool module
use rustfmt as imported_rustfmt;

// Standard library prelude
use Vec as ImportedVec;

// Built-in type
use u8 as imported_u8;

type A = imported_u8;

#[imported_inline]
#[imported_rustfmt::skip]
fn main() {
    imported_vec![0];
    ImportedVec::<u8>::new();
}
