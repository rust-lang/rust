//! This is a regression test for <https://github.com/rust-lang/rust/issues/137366>, ensuring
//! that we can use the `neon` target feature on ARM32 targets in rustdoc despite there
//! being a "forbidden" feature of the same name for aarch64, and rustdoc merging the
//! target features of all targets.
//@ check-pass
//@ revisions: arm aarch64
//@[arm] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[arm] needs-llvm-components: arm
//@[aarch64] compile-flags: --target aarch64-unknown-none-softfloat
//@[aarch64] needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![feature(arm_target_feature)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

// `fp-armv8` is "forbidden" on aarch64 as we tie it to `neon`.
#[target_feature(enable = "fp-armv8")]
pub fn fun1() {}

// This would usually be rejected as it changes the ABI.
// But we disable that check in rustdoc since we are building "for all targets" and the
// check can't really handle that.
#[target_feature(enable = "soft-float")]
pub fn fun2() {}
