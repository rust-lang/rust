//! Ensure ABI-incompatible features cannot be enabled via
//! `-Ctarget-feature`/`-Zllvm-target-feature`.
// These are just warnings for now.
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: x86 x86-llvm riscv riscv-llvm
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=+soft-float
//@[x86] needs-llvm-components: x86
//@[x86-llvm] compile-flags: --target=x86_64-unknown-linux-gnu -Zllvm-target-feature=+soft-float
//@[x86-llvm] needs-llvm-components: x86
//@[x86-llvm] ignore-backends: gcc
//@[riscv] compile-flags: --target=riscv32e-unknown-none-elf -Ctarget-feature=+d
//@[riscv] needs-llvm-components: riscv
//@[riscv-llvm] compile-flags: --target=riscv32e-unknown-none-elf -Zllvm-target-feature=+d
//@[riscv-llvm] needs-llvm-components: riscv
//@[riscv-llvm] ignore-backends: gcc

#![feature(no_core, riscv_target_feature)]
#![no_core]

// Cannot use minicore as some of these flags make it fail to compile, and it is those flags we are
// testing. The -Zllvm-target-feature flag is even a target modifier so we cannot compile minicore
// with different flags either.

//~? WARN must be disabled to ensure that the ABI of the current target can be implemented correctly
//[x86]~? WARN use a soft-float target instead
