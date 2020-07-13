// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

pub fn function_with_str<const STRING: &'static str>() -> &'static str {
    STRING
}

pub fn function_with_bytes<const BYTES: &'static [u8]>() -> &'static [u8] {
    BYTES
}

pub fn main() {
    assert_eq!(function_with_str::<"Rust">(), "Rust");
    assert_eq!(function_with_str::<"ℇ㇈↦">(), "ℇ㇈↦");
    assert_eq!(function_with_bytes::<b"AAAA">(), &[0x41, 0x41, 0x41, 0x41]);
    assert_eq!(function_with_bytes::<{&[0x41, 0x41, 0x41, 0x41]}>(), b"AAAA");
}
