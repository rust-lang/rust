//@ check-fail
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ add-core-stubs
//@ compile-flags: -O
// All architectures that have the attribute implemented:
//@ revisions: aarch64 powerpc s390x x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ [powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@ [powerpc] needs-llvm-components: powerpc
//@ [s390x] compile-flags: --target s390x-unknown-linux-gnu
//@ [s390x] needs-llvm-components: systemz
//@ [x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64] needs-llvm-components: x86
// FIXME: Add `xtensa` revision once `xtensa` support is fully available in upstream LLVM.

#![feature(no_core, rustc_attrs)]
#![crate_type = "lib"]
#![no_core]
extern crate minicore;
use minicore::*;

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
pub struct Type(usize);

#[rustc_abi(debug)]
pub extern "C" fn func(_: Type) {}
//~^ ERROR fn_abi_of(func) = FnAbi {
//~^^ ERROR mode: Indirect {
//~^^^ ERROR on_stack: false,
