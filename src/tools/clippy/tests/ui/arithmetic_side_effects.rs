#![allow(
    clippy::assign_op_pattern,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::no_effect,
    clippy::op_ref,
    clippy::unnecessary_owned_empty_strings,
    arithmetic_overflow,
    unconditional_panic
)]
#![feature(const_mut_refs, inline_const, saturating_int_impl)]
#![warn(clippy::arithmetic_side_effects)]

use core::num::{Saturating, Wrapping};

const ONE: i32 = 1;
const ZERO: i32 = 0;

#[derive(Clone, Copy)]
pub struct Custom;

macro_rules! impl_arith {
    ( $( $_trait:ident, $lhs:ty, $rhs:ty, $method:ident; )* ) => {
        $(
            impl core::ops::$_trait<$lhs> for $rhs {
                type Output = Custom;
                fn $method(self, _: $lhs) -> Self::Output { todo!() }
            }
        )*
    }
}

macro_rules! impl_assign_arith {
    ( $( $_trait:ident, $lhs:ty, $rhs:ty, $method:ident; )* ) => {
        $(
            impl core::ops::$_trait<$lhs> for $rhs {
                fn $method(&mut self, _: $lhs) {}
            }
        )*
    }
}

impl_arith!(
    Add, Custom, Custom, add;
    Div, Custom, Custom, div;
    Mul, Custom, Custom, mul;
    Rem, Custom, Custom, rem;
    Shl, Custom, Custom, shl;
    Shr, Custom, Custom, shr;
    Sub, Custom, Custom, sub;

    Add, Custom, &Custom, add;
    Div, Custom, &Custom, div;
    Mul, Custom, &Custom, mul;
    Rem, Custom, &Custom, rem;
    Shl, Custom, &Custom, shl;
    Shr, Custom, &Custom, shr;
    Sub, Custom, &Custom, sub;

    Add, &Custom, Custom, add;
    Div, &Custom, Custom, div;
    Mul, &Custom, Custom, mul;
    Rem, &Custom, Custom, rem;
    Shl, &Custom, Custom, shl;
    Shr, &Custom, Custom, shr;
    Sub, &Custom, Custom, sub;

    Add, &Custom, &Custom, add;
    Div, &Custom, &Custom, div;
    Mul, &Custom, &Custom, mul;
    Rem, &Custom, &Custom, rem;
    Shl, &Custom, &Custom, shl;
    Shr, &Custom, &Custom, shr;
    Sub, &Custom, &Custom, sub;
);

impl_assign_arith!(
    AddAssign, Custom, Custom, add_assign;
    DivAssign, Custom, Custom, div_assign;
    MulAssign, Custom, Custom, mul_assign;
    RemAssign, Custom, Custom, rem_assign;
    ShlAssign, Custom, Custom, shl_assign;
    ShrAssign, Custom, Custom, shr_assign;
    SubAssign, Custom, Custom, sub_assign;

    AddAssign, Custom, &Custom, add_assign;
    DivAssign, Custom, &Custom, div_assign;
    MulAssign, Custom, &Custom, mul_assign;
    RemAssign, Custom, &Custom, rem_assign;
    ShlAssign, Custom, &Custom, shl_assign;
    ShrAssign, Custom, &Custom, shr_assign;
    SubAssign, Custom, &Custom, sub_assign;

    AddAssign, &Custom, Custom, add_assign;
    DivAssign, &Custom, Custom, div_assign;
    MulAssign, &Custom, Custom, mul_assign;
    RemAssign, &Custom, Custom, rem_assign;
    ShlAssign, &Custom, Custom, shl_assign;
    ShrAssign, &Custom, Custom, shr_assign;
    SubAssign, &Custom, Custom, sub_assign;

    AddAssign, &Custom, &Custom, add_assign;
    DivAssign, &Custom, &Custom, div_assign;
    MulAssign, &Custom, &Custom, mul_assign;
    RemAssign, &Custom, &Custom, rem_assign;
    ShlAssign, &Custom, &Custom, shl_assign;
    ShrAssign, &Custom, &Custom, shr_assign;
    SubAssign, &Custom, &Custom, sub_assign;
);

impl core::ops::Neg for Custom {
    type Output = Custom;
    fn neg(self) -> Self::Output {
        todo!()
    }
}
impl core::ops::Neg for &Custom {
    type Output = Custom;
    fn neg(self) -> Self::Output {
        todo!()
    }
}

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

    const _: i32 = { let mut n = 1; n = -1; n = -(-1); n = -n; n };
    let _ = const { let mut n = 1; n = -1; n = -(-1); n = -n; n };
}

pub fn non_overflowing_ops_or_ops_already_handled_by_the_compiler_should_not_trigger_the_lint() {
    let mut _n = i32::MAX;

    // Assign
    _n += 0;
    _n += &0;
    _n -= 0;
    _n -= &0;
    _n += ZERO;
    _n += &ZERO;
    _n -= ZERO;
    _n -= &ZERO;
    _n /= 99;
    _n /= &99;
    _n %= 99;
    _n %= &99;
    _n *= 0;
    _n *= &0;
    _n *= 1;
    _n *= &1;
    _n *= ZERO;
    _n *= &ZERO;
    _n *= ONE;
    _n *= &ONE;
    _n += -0;
    _n += &-0;
    _n -= -0;
    _n -= &-0;
    _n += -ZERO;
    _n += &-ZERO;
    _n -= -ZERO;
    _n -= &-ZERO;
    _n /= -99;
    _n /= &-99;
    _n %= -99;
    _n %= &-99;
    _n *= -0;
    _n *= &-0;
    _n *= -1;
    _n *= &-1;

    // Binary
    _n = _n + 0;
    _n = _n + &0;
    _n = 0 + _n;
    _n = &0 + _n;
    _n = _n + ZERO;
    _n = _n + &ZERO;
    _n = ZERO + _n;
    _n = &ZERO + _n;
    _n = _n - 0;
    _n = _n - &0;
    _n = 0 - _n;
    _n = &0 - _n;
    _n = _n - ZERO;
    _n = _n - &ZERO;
    _n = ZERO - _n;
    _n = &ZERO - _n;
    _n = _n / 99;
    _n = _n / &99;
    _n = _n % 99;
    _n = _n % &99;
    _n = _n * 0;
    _n = _n * &0;
    _n = 0 * _n;
    _n = &0 * _n;
    _n = _n * 1;
    _n = _n * &1;
    _n = ZERO * _n;
    _n = &ZERO * _n;
    _n = _n * ONE;
    _n = _n * &ONE;
    _n = 1 * _n;
    _n = &1 * _n;
    _n = 23 + 85;

    // Unary
    _n = -2147483647;
    _n = -i32::MAX;
    _n = -i32::MIN;
    _n = -&2147483647;
    _n = -&i32::MAX;
    _n = -&i32::MIN;
}

pub fn unknown_ops_or_runtime_ops_that_can_overflow() {
    let mut _n = i32::MAX;
    let mut _custom = Custom;

    // Assign
    _n += 1;
    _n += &1;
    _n -= 1;
    _n -= &1;
    _n /= 0;
    _n /= &0;
    _n %= 0;
    _n %= &0;
    _n *= 2;
    _n *= &2;
    _n += -1;
    _n += &-1;
    _n -= -1;
    _n -= &-1;
    _n /= -0;
    _n /= &-0;
    _n %= -0;
    _n %= &-0;
    _n *= -2;
    _n *= &-2;
    _custom += Custom;
    _custom += &Custom;
    _custom -= Custom;
    _custom -= &Custom;
    _custom /= Custom;
    _custom /= &Custom;
    _custom %= Custom;
    _custom %= &Custom;
    _custom *= Custom;
    _custom *= &Custom;
    _custom >>= Custom;
    _custom >>= &Custom;
    _custom <<= Custom;
    _custom <<= &Custom;
    _custom += -Custom;
    _custom += &-Custom;
    _custom -= -Custom;
    _custom -= &-Custom;
    _custom /= -Custom;
    _custom /= &-Custom;
    _custom %= -Custom;
    _custom %= &-Custom;
    _custom *= -Custom;
    _custom *= &-Custom;
    _custom >>= -Custom;
    _custom >>= &-Custom;
    _custom <<= -Custom;
    _custom <<= &-Custom;

    // Binary
    _n = _n + 1;
    _n = _n + &1;
    _n = 1 + _n;
    _n = &1 + _n;
    _n = _n - 1;
    _n = _n - &1;
    _n = 1 - _n;
    _n = &1 - _n;
    _n = _n / 0;
    _n = _n / &0;
    _n = _n % 0;
    _n = _n % &0;
    _n = _n * 2;
    _n = _n * &2;
    _n = 2 * _n;
    _n = &2 * _n;
    _n = 23 + &85;
    _n = &23 + 85;
    _n = &23 + &85;
    _custom = _custom + _custom;
    _custom = _custom + &_custom;
    _custom = Custom + _custom;
    _custom = &Custom + _custom;
    _custom = _custom - Custom;
    _custom = _custom - &Custom;
    _custom = Custom - _custom;
    _custom = &Custom - _custom;
    _custom = _custom / Custom;
    _custom = _custom / &Custom;
    _custom = _custom % Custom;
    _custom = _custom % &Custom;
    _custom = _custom * Custom;
    _custom = _custom * &Custom;
    _custom = Custom * _custom;
    _custom = &Custom * _custom;
    _custom = Custom + &Custom;
    _custom = &Custom + Custom;
    _custom = &Custom + &Custom;
    _custom = _custom >> _custom;
    _custom = _custom >> &_custom;
    _custom = Custom << _custom;
    _custom = &Custom << _custom;

    // Unary
    _n = -_n;
    _n = -&_n;
    _custom = -_custom;
    _custom = -&_custom;
}

// Copied and pasted from the `integer_arithmetic` lint for comparison.
pub fn integer_arithmetic() {
    let mut i = 1i32;
    let mut var1 = 0i32;
    let mut var2 = -1i32;

    1 + i;
    i * 2;
    1 % i / 2;
    i - 2 + 2 - i;
    -i;
    i >> 1;
    i << 1;

    -1;
    -(-1);

    i & 1;
    i | 1;
    i ^ 1;

    i += 1;
    i -= 1;
    i *= 2;
    i /= 2;
    i /= 0;
    i /= -1;
    i /= var1;
    i /= var2;
    i %= 2;
    i %= 0;
    i %= -1;
    i %= var1;
    i %= var2;
    i <<= 3;
    i >>= 2;

    i |= 1;
    i &= 1;
    i ^= i;
}

fn main() {}
