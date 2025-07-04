//@ check-fail
//@ revisions: x86_64 aarch64 riscv64
//@ add-core-stubs
//@ [x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ [riscv64] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64] needs-llvm-components: riscv

#![feature(no_core, rustc_attrs)]
#![crate_type = "lib"]
#![no_core]
extern crate minicore;
use minicore::*;

#[rustc_pass_indirectly_in_non_rustic_abis]
//~^ ERROR: `#[rustc_pass_indirectly_in_non_rustic_abis]` can only be applied to `struct`s
//[riscv64]~^^ ERROR: support for `#[rustc_pass_indirectly_in_non_rustic_abis]` on `riscv64` has not yet been implemented
fn not_a_struct() {}

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
//[riscv64]~^ ERROR: support for `#[rustc_pass_indirectly_in_non_rustic_abis]` on `riscv64` has not yet been implemented
struct YesAStruct {
    foo: u8,
    bar: u16,
}
