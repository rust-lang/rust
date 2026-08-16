//! Verify the behavior differences between enabling and disabling overflow checks.

//@ run-pass
//@ needs-unwind
//@ revisions: ERROR WRAP
//@[ERROR] compile-flags: -C overflow-checks=true
//@[WRAP] compile-flags: -C overflow-checks=false

#![feature(cfg_overflow_checks)]

use std::hint::black_box as bb;
use std::{assert_matches, fmt, panic};

#[track_caller]
fn check<T: fmt::Debug + PartialEq>(func: fn() -> T, wrapping_res: T, name: &str) {
    let type_name = std::any::type_name::<T>();
    let res = panic::catch_unwind(func);
    if cfg!(overflow_checks) {
        assert_matches!(res, Err(_), "{type_name} {name}");
    } else {
        assert_eq!(res.unwrap(), wrapping_res, "{type_name} {name}");
    }
}

fn main() {
    check(|| bb(u32::MAX) + bb(1), 0, "add");
    check(|| bb(0u32) - bb(1), u32::MAX, "sub");
    check(|| bb(u32::MAX) * bb(2), u32::MAX << 1, "mul");
    check(|| bb(1u32) << bb(32), 1, "shl");
    check(|| bb(u32::MAX) >> bb(32), u32::MAX, "shr");
    check(|| bb(u32::MAX).pow(bb(2)), 1, "pow");
    check(|| bb(u32::MAX).next_power_of_two(), 0, "next_power_of_two");

    check(|| bb(i32::MAX) + bb(1), i32::MIN, "add");
    check(|| bb(i32::MIN) - bb(1), i32::MAX, "sub");
    check(|| bb(i32::MAX) * bb(2), i32::MAX << 1, "mul");
    check(|| -bb(i32::MIN), i32::MIN, "neg");
    check(|| bb(i32::MIN).abs(), i32::MIN, "abs");
    check(|| bb(1) << bb(32), 1, "shl");
    check(|| bb(i32::MAX) >> bb(32), i32::MAX, "shr");
    check(|| bb(i32::MAX).pow(bb(2)), 1, "pow");
}
