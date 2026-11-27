//@ add-minicore
//@ normalize-stderr: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ normalize-stderr: "(size): Size\([48] bytes\)" -> "$1: $$SOME_SIZE"
//@ normalize-stderr: "(can_unwind): (true|false)" -> "$1: $$SOME_BOOL"
//@ normalize-stderr: "(pointer is|valid_range:) 0\.\.=(4294967295|18446744073709551615)" -> "$1 $$FULL"
// This pattern is prepared for when we account for alignment in the niche.
//@ normalize-stderr: "(pointer is|valid_range:) [1-9]\.\.=(429496729[0-9]|1844674407370955161[0-9])" -> "$1 $$NON_NULL"
// Some attributes are only computed for release builds:
//@ compile-flags: -O
//@ revisions: generic win64
//@ [win64] compile-flags: --target x86_64-pc-windows-msvc
//@ [win64] needs-llvm-components: x86
//@ [generic] ignore-x86_64-pc-windows-gnu
//@ [generic] ignore-x86_64-pc-windows-gnullvm
//@ [generic] ignore-x86_64-pc-windows-msvc
//@ ignore-backends: gcc
#![feature(rustc_attrs)]
#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_abi(debug)]
fn u128_ret() -> u128 {
    //~^ ERROR: fn_abi
    1
}

#[rustc_abi(debug)]
fn scalar_pair_ret() -> (usize, usize) {
    //~^ ERROR: fn_abi
    (1, 2)
}

#[rustc_abi(debug)]
fn scalar_pair_ret2() -> (u32, u32) {
    //~^ ERROR: fn_abi
    (1, 2)
}

#[rustc_abi(debug)]
fn scalar_pair_ret3() -> (u128, u128) {
    //~^ ERROR: fn_abi
    (1, 2)
}
