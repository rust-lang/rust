//! Ensure ABI-required features cannot be disabled via `-Ctarget-feature`.
//! Also covers the case of a feature indirectly disabling another via feature implications.
//@ compile-flags: --crate-type=lib
//@ revisions: x86 x86-implied aarch64 riscv loongarch
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-x87
//@[x86] needs-llvm-components: x86
//@[x86-implied] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-sse
//@[x86-implied] needs-llvm-components: x86
//@[aarch64] compile-flags: --target=aarch64-unknown-linux-gnu -Ctarget-feature=-neon
//@[aarch64] needs-llvm-components: aarch64
//@[riscv] compile-flags: --target=riscv64gc-unknown-none-elf -Ctarget-feature=-d
//@[riscv] needs-llvm-components: riscv
//@[loongarch] compile-flags: --target=loongarch64-unknown-none -Ctarget-feature=-d
//@[loongarch] needs-llvm-components: loongarch
// For now this is just a warning.
//@ build-pass
// Remove some LLVM warnings that only show up sometimes.
//@ normalize-stderr: "\n[^\n]*(target-abi|lp64f)[^\n]*" -> ""

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized {}

//~? WARN must be enabled to ensure that the ABI of the current target can be implemented correctly
//[x86,riscv]~? WARN unstable feature specified for `-Ctarget-feature`
