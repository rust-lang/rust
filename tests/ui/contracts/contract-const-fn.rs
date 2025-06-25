//! Check if we can annotate a constant function with contracts.
//!
//! The contract is only checked at runtime, and it will not fail if evaluated statically.
//! This is an existing limitation due to the existing architecture and the lack of constant
//! closures.
//!
//@ revisions: all_pass runtime_fail_pre runtime_fail_post
//
//@ [all_pass] run-pass
//
//@ [runtime_fail_pre] run-crash
//@ [runtime_fail_post] run-crash
//
//@ [all_pass] compile-flags: -Zcontract-checks=yes
//@ [runtime_fail_pre] compile-flags: -Zcontract-checks=yes
//@ [runtime_fail_post] compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;
use core::contracts::*;

#[requires(x < 100)]
const fn less_than_100(x: u8) -> u8 {
    x
}

// This is wrong on purpose.
#[ensures(|ret| *ret)]
const fn always_true(b: bool) -> bool {
    b
}

const ZERO: u8 = less_than_100(0);
// This is no-op because the contract cannot be checked at compilation time.
const TWO_HUNDRED: u8 = less_than_100(200);

/// Example from <https://github.com/rust-lang/rust/issues/136925>.
#[ensures(move |ret: &u32| *ret > x)]
const fn broken_sum(x: u32, y: u32) -> u32 {
    x + y
}

fn main() {
    assert_eq!(ZERO, 0);
    assert_eq!(TWO_HUNDRED, 200);
    assert_eq!(broken_sum(0, 1), 1);
    assert_eq!(always_true(true), true);

    #[cfg(runtime_fail_post)]
    let _ok = always_true(false);

    // Runtime check should fail.
    #[cfg(runtime_fail_pre)]
    let _200 = less_than_100(200);
}
