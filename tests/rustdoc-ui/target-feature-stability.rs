//! This is a regression test for <https://github.com/rust-lang/rust/issues/137366>, ensuring
//! that we can use the `neon` target feature on ARM-32 targets in rustdoc despite there
//! being a "forbidden" feature of the same name for aarch64, and rustdoc merging the
//! target features of all targets.
//@ check-pass
//@ compile-flags: --target armv7-unknown-linux-gnueabihf

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![feature(arm_target_feature)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

// `fp-armv8` is "forbidden" on aarch64 as we tie it to `neon`.
#[target_feature(enable = "fp-armv8")]
pub fn fun() {}
