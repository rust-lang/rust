//@ compile-flags: -Clink-dead-code -Zinline-mir=no -Copt-level=0

// Matching against a constant array or slice of bytewise-comparable elements
// is lowered to a `PartialEq::eq` call that is assumed not to unwind. Check that
// the call resolves to the `BytewiseEq` specialisations, which compare the whole
// aggregate with an intrinsic, rather than to the generic element-by-element
// fallbacks, which would instantiate more items.

#![crate_type = "lib"]

const ARRAY: [u8; 4] = *b"abcd";
const SLICE: &[u8] = b"abcd";

//~ MONO_ITEM fn match_array
//~ MONO_ITEM fn std::array::equality::<impl std::cmp::PartialEq for [u8; 4]>::eq
//~ MONO_ITEM fn <u8 as std::array::equality::SpecArrayEq<u8, 4>>::spec_eq
pub fn match_array(x: &[u8; 4]) -> bool {
    matches!(*x, ARRAY)
}

//~ MONO_ITEM fn match_slice
//~ MONO_ITEM fn core::slice::cmp::<impl std::cmp::PartialEq for [u8]>::eq
//~ MONO_ITEM fn <u8 as core::slice::cmp::SlicePartialEq<u8>>::equal_same_length
pub fn match_slice(x: &[u8]) -> bool {
    matches!(x, SLICE)
}
