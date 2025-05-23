//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

pub trait T {
    #[target_feature(enable = "aes")]
    unsafe fn foo() {} //~^ERROR: should be applied to a function definition
}

impl T for i32 {
    #[target_feature(enable = "aes")]
    unsafe fn foo() {} //~^ERROR: should be applied to a function definition
}

struct S;
impl S {
    // Not a trait, no error.
    #[target_feature(enable = "aes")]
    unsafe fn foo() {}
}
