//@ run-pass

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

pub fn function_with_str<const STRING: &'static str>() -> &'static str {
    STRING
}

pub fn function_with_bytes<const BYTES: &'static [u8]>() -> &'static [u8] {
    BYTES
}

// Also check the codepaths for custom DST
#[derive(std::marker::ConstParamTy, PartialEq, Eq)]
struct MyStr(str);

fn function_with_my_str<const S: &'static MyStr>() -> &'static MyStr {
    S
}

impl MyStr {
    const fn new(s: &'static str) -> &'static MyStr {
        unsafe { std::mem::transmute(s) }
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

pub fn main() {
    assert_eq!(function_with_str::<"Rust">(), "Rust");
    assert_eq!(function_with_str::<"ℇ㇈↦">(), "ℇ㇈↦");
    assert_eq!(function_with_bytes::<b"AAAA">(), &[0x41, 0x41, 0x41, 0x41]);
    assert_eq!(function_with_bytes::<{ &[0x41, 0x41, 0x41, 0x41] }>(), b"AAAA");

    assert_eq!(function_with_my_str::<{ MyStr::new("hello") }>().as_str(), "hello");
}
