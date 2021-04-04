//[full] run-pass
// revisions: min full

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub fn function_with_str<const STRING: &'static str>() -> &'static str {
    //[min]~^ ERROR `&'static str` is forbidden
    STRING
}

pub fn function_with_bytes<const BYTES: &'static [u8]>() -> &'static [u8] {
    //[min]~^ ERROR `&'static [u8]` is forbidden
    BYTES
}

pub fn main() {
    assert_eq!(function_with_str::<"Rust">(), "Rust");
    assert_eq!(function_with_str::<"ℇ㇈↦">(), "ℇ㇈↦");
    assert_eq!(function_with_bytes::<b"AAAA">(), &[0x41, 0x41, 0x41, 0x41]);
    assert_eq!(function_with_bytes::<{&[0x41, 0x41, 0x41, 0x41]}>(), b"AAAA");
}
