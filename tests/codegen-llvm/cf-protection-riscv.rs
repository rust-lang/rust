// Test that the RISC-V specific module flag and function attribute are emitted with different
// control-flow protection flags.

//@ add-minicore
//@ revisions: none branch return_ full branch_llvm23
//@ needs-llvm-components: riscv
//@ [none] compile-flags: -Z cf-protection=none
//@ [branch] compile-flags: -Z cf-protection=branch
//@ [return_] compile-flags: -Z cf-protection=return -Ctarget-feature=+zimop
//@ [full] compile-flags: -Z cf-protection=full -Ctarget-feature=+zimop
// Before LLVM 24, Zicfilp has to be enabled so that tail calls use `t2`.
//@ [branch_llvm23] compile-flags: -Z cf-protection=branch
//@ [branch_llvm23] max-llvm-major-version: 23
//@ compile-flags: --target riscv64gc-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: define{{.*}}void @test(){{.*}} #[[ATTR:[0-9]+]]
#[no_mangle]
pub fn test() {}

// none-NOT: hw-shadow-stack
// none-NOT: zicfilp
// branch-NOT: hw-shadow-stack
// return_: attributes #[[ATTR]] = {{.*}}"hw-shadow-stack"
// return_-NOT: zicfilp
// full: attributes #[[ATTR]] = {{.*}}"hw-shadow-stack"
// branch_llvm23: attributes #[[ATTR]] = {{.*}}"target-features"="{{[^"]*}}+experimental-zicfilp

// none-NOT: !"cf-branch-label-scheme"
// return_-NOT: !"cf-branch-label-scheme"
// branch: !"cf-branch-label-scheme", !"unlabeled"
// full: !"cf-branch-label-scheme", !"unlabeled"
