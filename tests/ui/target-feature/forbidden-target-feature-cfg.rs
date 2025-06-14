//! Ensure "forbidden" target features are not exposed via `cfg`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
//@ check-pass
#![feature(no_core, lang_items)]
#![no_core]
#![allow(unexpected_cfgs)]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

// The compile_error macro does not exist, so if the `cfg` evaluates to `true` this
// complains about the missing macro rather than showing the error... but that's good enough.
#[cfg(target_feature = "forced-atomics")]
compile_error!("the forced-atomics feature should not be exposed in `cfg`");
