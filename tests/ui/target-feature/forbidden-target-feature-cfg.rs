//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ check-pass
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![allow(unexpected_cfgs)]

#[lang = "sized"]
pub trait Sized {}

// The compile_error macro does not exist, so if the `cfg` evaluates to `true` this
// complains about the missing macro rather than showing the error... but that's good enough.
#[cfg(target_feature = "soft-float")]
compile_error!("the soft-float feature should not be exposed in `cfg`");
