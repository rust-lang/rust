//@ add-minicore
//@ revisions: return_rv64 full_rv32 return_zimop return_attr branch_rv32 full_rva23
//@ compile-flags: --crate-type=rlib
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

//@ [return_rv64] compile-flags: --target=riscv64gc-unknown-none-elf -Zcf-protection=return
//@ [return_rv64] build-fail
//@ [return_rv64] minicore-compile-flags: -Ctarget-feature=+zimop

//@ [full_rv32] compile-flags: --target=riscv32imc-unknown-none-elf -Zcf-protection=full
//@ [full_rv32] build-fail
//@ [full_rv32] minicore-compile-flags: -Ctarget-feature=+zimop

//@ [return_zimop] compile-flags: --target=riscv64gc-unknown-none-elf -Zcf-protection=return
//@ [return_zimop] compile-flags: -Ctarget-feature=+zimop
//@ [return_zimop] build-pass

//@ [return_attr] compile-flags: --target=riscv64gc-unknown-none-elf -Zcf-protection=return
//@ [return_attr] build-pass
//@ [return_attr] minicore-compile-flags: -Ctarget-feature=+zimop

//@ [branch_rv32] compile-flags: --target=riscv32imc-unknown-none-elf -Zcf-protection=branch
//@ [branch_rv32] build-pass

//@ [full_rva23] compile-flags: --target=riscv64a23-unknown-linux-gnu -Zcf-protection=full
//@ [full_rva23] build-pass

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
#[cfg_attr(return_attr, target_feature(enable = "zimop"))]
pub fn test() {}

//[return_rv64]~? ERROR `-Zcf-protection=return` and `full` require `-Ctarget-feature=+zimop` on RISC-V
//[full_rv32]~? ERROR `-Zcf-protection=return` and `full` require `-Ctarget-feature=+zimop` on RISC-V
