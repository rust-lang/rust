// Regression test for issue https://github.com/rust-lang/rust/issues/143987
// Ensure that using `#[align]` on struct fields produces an error
// instead of causing an ICE (Internal Compiler Error)

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

struct Data {
    #[rustc_align(8)] //~ ERROR `#[rustc_align]` attribute cannot be used on struct fields
    x: usize,
}

// Test with invalid type to match the original issue more closely
struct DataInvalid {
    #[rustc_align(8)] //~ ERROR `#[rustc_align]` attribute cannot be used on struct fields
    x: usize8, //~ ERROR cannot find type `usize8` in this scope
}

// Test with tuple struct
struct TupleData(
    #[rustc_align(32)] //~ ERROR `#[rustc_align]` attribute cannot be used on struct fields
    u32
);

// Test that it works correctly on functions (no error)
#[rustc_align(16)]
fn aligned_function() {}

fn main() {}
