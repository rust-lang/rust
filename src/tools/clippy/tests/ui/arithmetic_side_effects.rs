//@aux-build:proc_macro_derive.rs

#![feature(f128)]
#![feature(f16)]
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
#![warn(clippy::arithmetic_side_effects)]

extern crate proc_macro_derive;

use core::num::{NonZero, Saturating, Wrapping};

const ONE: i32 = 1;
const ZERO: i32 = 0;

#[derive(Clone, Copy)]
pub struct Custom;

#[derive(proc_macro_derive::ShadowDerive)]
pub struct Nothing;

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
    let _ = 1f16 + 1f16;
    //~^ arithmetic_side_effects
    let _ = 1f32 + 1f32;
    let _ = 1f64 + 1f64;
    let _ = 1f128 + 1f128;
    //~^ arithmetic_side_effects

    let _ = Saturating(0u32) + Saturating(0u32);
    let _ = String::new() + "";
    let _ = String::new() + &String::new();
    //~^ arithmetic_side_effects
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

    // Method
    _n.saturating_div(1);
    _n.wrapping_div(1);
    _n.wrapping_rem(1);
    _n.wrapping_rem_euclid(1);

    _n.saturating_div(1);
    _n.checked_div(1);
    _n.checked_rem(1);
    _n.checked_rem_euclid(1);

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
    //~^ arithmetic_side_effects
    _n += &1;
    //~^ arithmetic_side_effects
    _n -= 1;
    //~^ arithmetic_side_effects
    _n -= &1;
    //~^ arithmetic_side_effects
    _n /= 0;
    //~^ arithmetic_side_effects
    _n /= &0;
    //~^ arithmetic_side_effects
    _n %= 0;
    //~^ arithmetic_side_effects
    _n %= &0;
    //~^ arithmetic_side_effects
    _n *= 2;
    //~^ arithmetic_side_effects
    _n *= &2;
    //~^ arithmetic_side_effects
    _n += -1;
    //~^ arithmetic_side_effects
    _n += &-1;
    //~^ arithmetic_side_effects
    _n -= -1;
    //~^ arithmetic_side_effects
    _n -= &-1;
    //~^ arithmetic_side_effects
    _n /= -0;
    //~^ arithmetic_side_effects
    _n /= &-0;
    //~^ arithmetic_side_effects
    _n %= -0;
    //~^ arithmetic_side_effects
    _n %= &-0;
    //~^ arithmetic_side_effects
    _n *= -2;
    //~^ arithmetic_side_effects
    _n *= &-2;
    //~^ arithmetic_side_effects
    _custom += Custom;
    //~^ arithmetic_side_effects
    _custom += &Custom;
    //~^ arithmetic_side_effects
    _custom -= Custom;
    //~^ arithmetic_side_effects
    _custom -= &Custom;
    //~^ arithmetic_side_effects
    _custom /= Custom;
    //~^ arithmetic_side_effects
    _custom /= &Custom;
    //~^ arithmetic_side_effects
    _custom %= Custom;
    //~^ arithmetic_side_effects
    _custom %= &Custom;
    //~^ arithmetic_side_effects
    _custom *= Custom;
    //~^ arithmetic_side_effects
    _custom *= &Custom;
    //~^ arithmetic_side_effects
    _custom >>= Custom;
    //~^ arithmetic_side_effects
    _custom >>= &Custom;
    //~^ arithmetic_side_effects
    _custom <<= Custom;
    //~^ arithmetic_side_effects
    _custom <<= &Custom;
    //~^ arithmetic_side_effects
    _custom += -Custom;
    //~^ arithmetic_side_effects
    _custom += &-Custom;
    //~^ arithmetic_side_effects
    _custom -= -Custom;
    //~^ arithmetic_side_effects
    _custom -= &-Custom;
    //~^ arithmetic_side_effects
    _custom /= -Custom;
    //~^ arithmetic_side_effects
    _custom /= &-Custom;
    //~^ arithmetic_side_effects
    _custom %= -Custom;
    //~^ arithmetic_side_effects
    _custom %= &-Custom;
    //~^ arithmetic_side_effects
    _custom *= -Custom;
    //~^ arithmetic_side_effects
    _custom *= &-Custom;
    //~^ arithmetic_side_effects
    _custom >>= -Custom;
    //~^ arithmetic_side_effects
    _custom >>= &-Custom;
    //~^ arithmetic_side_effects
    _custom <<= -Custom;
    //~^ arithmetic_side_effects
    _custom <<= &-Custom;
    //~^ arithmetic_side_effects

    // Binary
    _n = _n + 1;
    //~^ arithmetic_side_effects
    _n = _n + &1;
    //~^ arithmetic_side_effects
    _n = 1 + _n;
    //~^ arithmetic_side_effects
    _n = &1 + _n;
    //~^ arithmetic_side_effects
    _n = _n - 1;
    //~^ arithmetic_side_effects
    _n = _n - &1;
    //~^ arithmetic_side_effects
    _n = 1 - _n;
    //~^ arithmetic_side_effects
    _n = &1 - _n;
    //~^ arithmetic_side_effects
    _n = _n / 0;
    //~^ arithmetic_side_effects
    _n = _n / &0;
    //~^ arithmetic_side_effects
    _n = _n % 0;
    //~^ arithmetic_side_effects
    _n = _n % &0;
    //~^ arithmetic_side_effects
    _n = _n * 2;
    //~^ arithmetic_side_effects
    _n = _n * &2;
    //~^ arithmetic_side_effects
    _n = 2 * _n;
    //~^ arithmetic_side_effects
    _n = &2 * _n;
    //~^ arithmetic_side_effects
    _n = 23 + &85;
    //~^ arithmetic_side_effects
    _n = &23 + 85;
    //~^ arithmetic_side_effects
    _n = &23 + &85;
    //~^ arithmetic_side_effects
    _custom = _custom + _custom;
    //~^ arithmetic_side_effects
    _custom = _custom + &_custom;
    //~^ arithmetic_side_effects
    _custom = Custom + _custom;
    //~^ arithmetic_side_effects
    _custom = &Custom + _custom;
    //~^ arithmetic_side_effects
    _custom = _custom - Custom;
    //~^ arithmetic_side_effects
    _custom = _custom - &Custom;
    //~^ arithmetic_side_effects
    _custom = Custom - _custom;
    //~^ arithmetic_side_effects
    _custom = &Custom - _custom;
    //~^ arithmetic_side_effects
    _custom = _custom / Custom;
    //~^ arithmetic_side_effects
    _custom = _custom / &Custom;
    //~^ arithmetic_side_effects
    _custom = _custom % Custom;
    //~^ arithmetic_side_effects
    _custom = _custom % &Custom;
    //~^ arithmetic_side_effects
    _custom = _custom * Custom;
    //~^ arithmetic_side_effects
    _custom = _custom * &Custom;
    //~^ arithmetic_side_effects
    _custom = Custom * _custom;
    //~^ arithmetic_side_effects
    _custom = &Custom * _custom;
    //~^ arithmetic_side_effects
    _custom = Custom + &Custom;
    //~^ arithmetic_side_effects
    _custom = &Custom + Custom;
    //~^ arithmetic_side_effects
    _custom = &Custom + &Custom;
    //~^ arithmetic_side_effects
    _custom = _custom >> _custom;
    //~^ arithmetic_side_effects
    _custom = _custom >> &_custom;
    //~^ arithmetic_side_effects
    _custom = Custom << _custom;
    //~^ arithmetic_side_effects
    _custom = &Custom << _custom;
    //~^ arithmetic_side_effects

    // Method
    _n.saturating_div(0);
    //~^ arithmetic_side_effects
    _n.wrapping_div(0);
    //~^ arithmetic_side_effects
    _n.wrapping_rem(0);
    //~^ arithmetic_side_effects
    _n.wrapping_rem_euclid(0);
    //~^ arithmetic_side_effects

    _n.saturating_div(_n);
    //~^ arithmetic_side_effects
    _n.wrapping_div(_n);
    //~^ arithmetic_side_effects
    _n.wrapping_rem(_n);
    //~^ arithmetic_side_effects
    _n.wrapping_rem_euclid(_n);
    //~^ arithmetic_side_effects

    _n.saturating_div(*Box::new(_n));
    //~^ arithmetic_side_effects

    // Unary
    _n = -_n;
    //~^ arithmetic_side_effects
    _n = -&_n;
    //~^ arithmetic_side_effects
    _custom = -_custom;
    //~^ arithmetic_side_effects
    _custom = -&_custom;
    //~^ arithmetic_side_effects
    _ = -*Box::new(_n);
    //~^ arithmetic_side_effects
}

// Copied and pasted from the `integer_arithmetic` lint for comparison.
pub fn integer_arithmetic() {
    let mut i = 1i32;
    let mut var1 = 0i32;
    let mut var2 = -1i32;

    1 + i;
    //~^ arithmetic_side_effects
    i * 2;
    //~^ arithmetic_side_effects
    1 % i / 2;
    //~^ arithmetic_side_effects
    i - 2 + 2 - i;
    //~^ arithmetic_side_effects
    -i;
    //~^ arithmetic_side_effects
    i >> 1;
    i << 1;

    -1;
    -(-1);

    i & 1;
    i | 1;
    i ^ 1;

    i += 1;
    //~^ arithmetic_side_effects
    i -= 1;
    //~^ arithmetic_side_effects
    i *= 2;
    //~^ arithmetic_side_effects
    i /= 2;
    i /= 0;
    //~^ arithmetic_side_effects
    i /= -1;
    i /= var1;
    //~^ arithmetic_side_effects
    i /= var2;
    //~^ arithmetic_side_effects
    i %= 2;
    i %= 0;
    //~^ arithmetic_side_effects
    i %= -1;
    i %= var1;
    //~^ arithmetic_side_effects
    i %= var2;
    //~^ arithmetic_side_effects
    i <<= 3;
    i >>= 2;

    i |= 1;
    i &= 1;
    i ^= i;
}

pub fn issue_10583(a: u16) -> u16 {
    10 / a
    //~^ arithmetic_side_effects
}

pub fn issue_10767() {
    let n = &1.0;
    n + n;
    3.1_f32 + &1.2_f32;
    &3.4_f32 + 1.5_f32;
    &3.5_f32 + &1.3_f32;
}

pub fn issue_10792() {
    struct One {
        a: u32,
    }
    struct Two {
        b: u32,
        c: u64,
    }
    const ONE: One = One { a: 1 };
    const TWO: Two = Two { b: 2, c: 3 };
    let _ = 10 / ONE.a;
    let _ = 10 / TWO.b;
    let _ = 10 / TWO.c;
}

pub fn issue_11145() {
    let mut x: Wrapping<u32> = Wrapping(0_u32);
    x += 1;
}

pub fn issue_11262() {
    let one = 1;
    let zero = 0;
    let _ = 2 / one;
    let _ = 2 / zero;
}

pub fn issue_11392() {
    fn example_div(unsigned: usize, nonzero_unsigned: NonZero<usize>) -> usize {
        unsigned / nonzero_unsigned
    }

    fn example_rem(unsigned: usize, nonzero_unsigned: NonZero<usize>) -> usize {
        unsigned % nonzero_unsigned
    }

    let (unsigned, nonzero_unsigned) = (0, NonZero::new(1).unwrap());
    example_div(unsigned, nonzero_unsigned);
    example_rem(unsigned, nonzero_unsigned);
}

pub fn issue_11393() {
    fn example_div(x: Wrapping<i32>, maybe_zero: Wrapping<i32>) -> Wrapping<i32> {
        x / maybe_zero
        //~^ arithmetic_side_effects
    }

    fn example_rem(x: Wrapping<i32>, maybe_zero: Wrapping<i32>) -> Wrapping<i32> {
        x % maybe_zero
        //~^ arithmetic_side_effects
    }

    let [x, maybe_zero] = [1, 0].map(Wrapping);
    example_div(x, maybe_zero);
    example_rem(x, maybe_zero);
}

pub fn issue_12318() {
    use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
    let mut one: i32 = 1;
    one.add_assign(1);
    //~^ arithmetic_side_effects
    one.div_assign(1);
    one.mul_assign(1);
    one.rem_assign(1);
    one.sub_assign(1);
    //~^ arithmetic_side_effects
}

pub fn issue_15225() {
    use core::num::{NonZero, NonZeroU8};

    let one = const { NonZeroU8::new(1).unwrap() };
    let _ = one.get() - 1;

    let one: NonZero<u8> = const { NonZero::new(1).unwrap() };
    let _ = one.get() - 1;

    type AliasedType = u8;
    let one: NonZero<AliasedType> = const { NonZero::new(1).unwrap() };
    let _ = one.get() - 1;
}

pub fn explicit_methods() {
    use core::ops::Add;
    let one: i32 = 1;
    one.add(&one);
    //~^ arithmetic_side_effects
    Box::new(one).add(one);
    //~^ arithmetic_side_effects
}

fn main() {}
