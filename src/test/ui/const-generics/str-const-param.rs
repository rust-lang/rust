// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub fn function_with_str<const STRING: &'static str>() -> &'static str {
    STRING
}

pub fn main() {
    assert_eq!(function_with_str::<"Rust">(), "Rust");
}
