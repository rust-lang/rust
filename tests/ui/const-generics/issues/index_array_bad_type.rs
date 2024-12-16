//@ check-fail
//@ compile-flags: -C opt-level=0

#![crate_type = "lib"]

// This used to fail in the known-panics lint, as the MIR was ill-typed due to
// the length constant not actually having type usize.
// https://github.com/rust-lang/rust/issues/134352

pub struct BadStruct<const N: i64>(pub [u8; N]);
//~^ ERROR: the constant `N` is not of type `usize`

pub fn bad_array_length_type(value: BadStruct<3>) -> u8 {
    value.0[0]
}
