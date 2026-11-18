//! Ensure ABI-required features cannot be disabled via `-Ctarget-feature`.
//! Also covers the case of a feature indirectly disabling another via feature implications.
//@ compile-flags: --crate-type=lib
//@ revisions: x86 x86-implied aarch64 powerpc64 powerpc64le riscv loongarch
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-x87
//@[x86] needs-llvm-components: x86
//@[x86] build-pass
//@[x86-implied] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-sse
//@[x86-implied] needs-llvm-components: x86
//@[x86-implied] build-pass
//@[powerpc64] compile-flags: --target=powerpc64-unknown-linux-musl -Ctarget-feature=-altivec
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64] build-pass
//@[powerpc64le] compile-flags: --target=powerpc64le-unknown-linux-gnu -Ctarget-feature=-altivec
//@[powerpc64le] needs-llvm-components: powerpc
//@[powerpc64le] build-pass
//@[riscv] compile-flags: --target=riscv64gc-unknown-none-elf -Ctarget-feature=-d
//@[riscv] needs-llvm-components: riscv
//@[riscv] build-pass
//@[loongarch] compile-flags: --target=loongarch64-unknown-none -Ctarget-feature=-d
//@[loongarch] needs-llvm-components: loongarch
//@[loongarch] build-pass

//@[aarch64] compile-flags: --target=aarch64-unknown-linux-gnu -Ctarget-feature=-neon
//@[aarch64] needs-llvm-components: aarch64
//@[aarch64] minicore-compile-flags: -Ctarget-feature=+neon
// Just a warning on most targets, but a hard error on aarch64
//@[aarch64] check-fail

// Remove some LLVM warnings that only show up sometimes.
//@ normalize-stderr: "\n[^\n]*(target-abi|lp64f)[^\n]*" -> ""
//@ add-minicore

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//[x86,x86-implied,powerpc64,powerpc64le,riscv,loongarch]~? WARN must be enabled to ensure that the ABI
//[aarch64]~? ERROR must be enabled to ensure that the ABI
