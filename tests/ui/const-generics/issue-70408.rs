//@ build-pass

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

pub fn function_with_bytes<const BYTES: &'static [u8; 4]>() -> &'static [u8] {
    BYTES
}

pub fn main() {
    assert_eq!(function_with_bytes::<b"AAAA">(), &[0x41, 0x41, 0x41, 0x41]);
    assert_eq!(function_with_bytes::<{ &[0x41, 0x41, 0x41, 0x41] }>(), b"AAAA");
}
