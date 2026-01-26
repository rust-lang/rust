//! The soft-float target feature is *not* exposed as `cfg` on x86.
//@ revisions: soft hard
//@[hard] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@[hard] needs-llvm-components: x86
//@[soft] compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@[soft] needs-llvm-components: x86
//@ check-pass
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core)]
#![no_core]
#![allow(unexpected_cfgs)]

// The compile_error macro does not exist, so if the `cfg` evaluates to `true` this
// complains about the missing macro rather than showing the error... but that's good enough.
#[cfg(target_feature = "soft-float")]
compile_error!("the soft-float feature should NOT be exposed in `cfg`");
