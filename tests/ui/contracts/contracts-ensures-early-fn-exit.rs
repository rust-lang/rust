//@ revisions: unchk_pass chk_pass chk_fail_try chk_fail_ret chk_fail_yeet
//
//@ [unchk_pass] run-pass
//@ [chk_pass] run-pass
//@ [chk_fail_try] run-crash
//@ [chk_fail_ret] run-crash
//@ [chk_fail_yeet] run-crash
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_try] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_ret] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_yeet] compile-flags: -Zcontract-checks=yes
//! This test ensures that ensures clauses are checked for different return points of a function.

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
#![feature(yeet_expr)]

/// This ensures will fail in different return points depending on the input.
#[core::contracts::ensures(|ret: &Option<u32>| ret.is_some())]
fn try_sum(x: u32, y: u32, z: u32) -> Option<u32> {
    // Use Yeet to return early.
    if x == u32::MAX && (y > 0 || z > 0) { do yeet }

    // Use `?` to early return.
    let partial = x.checked_add(y)?;

    // Explicitly use `return` clause.
    if u32::MAX - partial < z {
        return None;
    }

    Some(partial + z)
}

fn main() {
    // This should always succeed
    assert_eq!(try_sum(0, 1, 2), Some(3));

    #[cfg(any(unchk_pass, chk_fail_yeet))]
    assert_eq!(try_sum(u32::MAX, 1, 1), None);

    #[cfg(any(unchk_pass, chk_fail_try))]
    assert_eq!(try_sum(u32::MAX - 10, 12, 0), None);

    #[cfg(any(unchk_pass, chk_fail_ret))]
    assert_eq!(try_sum(u32::MAX - 10, 2, 100), None);
}
