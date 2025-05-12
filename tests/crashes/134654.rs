//@ known-bug: #134654
//@ compile-flags: -Zmir-enable-passes=+GVN -Zmir-enable-passes=+Inline -Zvalidate-mir
//@ only-x86_64

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

fn function_with_bytes<const BYTES:
    &'static [u8; 0xa9008fb6c9d81e42_0e25730562a601c8_u128]>() -> &'static [u8] {
    BYTES
}

fn main() {
    function_with_bytes::<b"aa">() == &[];
}
