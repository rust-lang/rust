#![allow(
    clippy::assign_op_pattern,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::unnecessary_owned_empty_strings,
    arithmetic_overflow,
    unconditional_panic
)]
#![feature(inline_const, saturating_int_impl)]
#![warn(clippy::arithmetic_side_effects)]

use core::num::{Saturating, Wrapping};

pub fn association_with_structures_should_not_trigger_the_lint() {
    enum Foo {
        Bar = -2,
    }

    impl Trait for Foo {
        const ASSOC: i32 = {
            let _: [i32; 1 + 1];
            fn foo() {}
            1 + 1
        };
    }

    struct Baz([i32; 1 + 1]);

    trait Trait {
        const ASSOC: i32 = 1 + 1;
    }

    type Alias = [i32; 1 + 1];

    union Qux {
        field: [i32; 1 + 1],
    }

    let _: [i32; 1 + 1] = [0, 0];

    let _: [i32; 1 + 1] = {
        let a: [i32; 1 + 1] = [0, 0];
        a
    };
}

pub fn hard_coded_allowed() {
    let _ = 1f32 + 1f32;
    let _ = 1f64 + 1f64;

    let _ = Saturating(0u32) + Saturating(0u32);
    let _ = String::new() + "";
    let _ = Wrapping(0u32) + Wrapping(0u32);

    let saturating: Saturating<u32> = Saturating(0u32);
    let string: String = String::new();
    let wrapping: Wrapping<u32> = Wrapping(0u32);

    let inferred_saturating = saturating + saturating;
    let inferred_string = string + "";
    let inferred_wrapping = wrapping + wrapping;

    let _ = inferred_saturating + inferred_saturating;
    let _ = inferred_string + "";
    let _ = inferred_wrapping + inferred_wrapping;
}

#[rustfmt::skip]
pub fn const_ops_should_not_trigger_the_lint() {
    const _: i32 = { let mut n = 1; n += 1; n };
    let _ = const { let mut n = 1; n += 1; n };

    const _: i32 = { let mut n = 1; n = n + 1; n };
    let _ = const { let mut n = 1; n = n + 1; n };

    const _: i32 = { let mut n = 1; n = 1 + n; n };
    let _ = const { let mut n = 1; n = 1 + n; n };

    const _: i32 = 1 + 1;
    let _ = const { 1 + 1 };

    const _: i32 = { let mut n = -1; n = -(-1); n = -n; n };
    let _ = const { let mut n = -1; n = -(-1); n = -n; n };
}

pub fn non_overflowing_runtime_ops_or_ops_already_handled_by_the_compiler() {
    let mut _n = i32::MAX;

    // Assign
    _n += 0;
    _n -= 0;
    _n /= 99;
    _n %= 99;
    _n *= 0;
    _n *= 1;

    // Binary
    _n = _n + 0;
    _n = 0 + _n;
    _n = _n - 0;
    _n = 0 - _n;
    _n = _n / 99;
    _n = _n % 99;
    _n = _n * 0;
    _n = 0 * _n;
    _n = _n * 1;
    _n = 1 * _n;
    _n = 23 + 85;

    // Unary
    _n = -1;
    _n = -(-1);
}

pub fn overflowing_runtime_ops() {
    let mut _n = i32::MAX;

    // Assign
    _n += 1;
    _n -= 1;
    _n /= 0;
    _n %= 0;
    _n *= 2;

    // Binary
    _n = _n + 1;
    _n = 1 + _n;
    _n = _n - 1;
    _n = 1 - _n;
    _n = _n / 0;
    _n = _n % 0;
    _n = _n * 2;
    _n = 2 * _n;

    // Unary
    _n = -_n;
}

fn main() {}
