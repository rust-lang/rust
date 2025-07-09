//! Test that contracts can be applied to generic functions.

//@ revisions: unchk_pass chk_pass chk_fail_pre chk_fail_post chk_const_fail
//
//@ [unchk_pass] run-pass
//@ [chk_pass] run-pass
//
//@ [chk_fail_pre] run-crash
//@ [chk_fail_post] run-crash
//@ [chk_const_fail] run-crash
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_pre] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_post] compile-flags: -Zcontract-checks=yes
//@ [chk_const_fail] compile-flags: -Zcontract-checks=yes

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

use std::ops::Sub;

/// Dummy fn contract that precondition fails for val < 0, and post-condition fail for val == 1
#[core::contracts::requires(val > 0u8.into())]
#[core::contracts::ensures(|ret| *ret > 0u8.into())]
fn decrement<T>(val: T) -> T
where T: PartialOrd + Sub<Output=T> + From<u8>
{
    val - 1u8.into()
}

/// Create a structure that takes a constant parameter.
#[allow(dead_code)]
struct Capped<const MAX: usize>(usize);

/// Now declare a function to create stars which shouldn't exceed 5 stars.
// Add redundant braces to ensure the built-in macro can handle this syntax.
#[allow(unused_braces)]
#[core::contracts::requires(num <= 5)]
unsafe fn stars_unchecked(num: usize) -> Capped<{ 5 }> {
    Capped(num)
}


fn main() {
    check_decrement();
    check_stars();
}

fn check_stars() {
    // This should always pass.
    let _ = unsafe { stars_unchecked(3) };

    // This violates the contract.
    #[cfg(any(unchk_pass, chk_const_fail))]
    let _ = unsafe { stars_unchecked(10) };
}

fn check_decrement() {
    // This should always pass
    assert_eq!(decrement(10u8), 9u8);

    // This should fail requires but pass with no contract check.
    #[cfg(any(unchk_pass, chk_fail_pre))]
    assert_eq!(decrement(-2i128), -3i128);

    // This should fail ensures but pass with no contract check.
    #[cfg(any(unchk_pass, chk_fail_post))]
    assert_eq!(decrement(1i32), 0i32);
}
