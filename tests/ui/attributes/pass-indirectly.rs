//@ check-fail
//@ revisions: x86_64 aarch64
//@ add-core-stubs
//@ [x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64

#![feature(no_core, rustc_attrs)]
#![crate_type = "lib"]
#![no_core]
extern crate minicore;
use minicore::*;

#[rustc_pass_indirectly_in_non_rustic_abis]
//~^ ERROR: `#[rustc_pass_indirectly_in_non_rustic_abis]` can only be applied to `struct`s
fn not_a_struct() {}

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
struct YesAStruct {
    foo: u8,
    bar: u16,
}
