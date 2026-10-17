//@aux-build:proc_macros.rs

#![warn(clippy::needless_nonzero_get)]
// the `msrv` functions below deliberately use items newer than their MSRV
#![allow(clippy::incompatible_msrv)]

extern crate proc_macros;

use proc_macros::with_span;
use std::num::{NonZero, NonZeroI32, NonZeroU32};

fn unsigned(nz: NonZero<u32>) {
    let _ = nz.get().leading_zeros();
    //~^ needless_nonzero_get
    let _ = nz.get().trailing_zeros();
    //~^ needless_nonzero_get
    let _ = nz.get().is_power_of_two();
    //~^ needless_nonzero_get
    let _ = nz.get().ilog2();
    //~^ needless_nonzero_get
    let _ = nz.get().ilog10();
    //~^ needless_nonzero_get
}

fn signed(nz: NonZero<i32>) {
    let _ = nz.get().leading_zeros();
    //~^ needless_nonzero_get
    let _ = nz.get().trailing_zeros();
    //~^ needless_nonzero_get
    let _ = nz.get().is_positive();
    //~^ needless_nonzero_get
    let _ = nz.get().is_negative();
    //~^ needless_nonzero_get
}

fn unsigned_widths(a: NonZero<u8>, b: NonZero<u16>, c: NonZero<u64>, d: NonZero<u128>, e: NonZero<usize>) {
    let _ = a.get().leading_zeros();
    //~^ needless_nonzero_get
    let _ = b.get().trailing_zeros();
    //~^ needless_nonzero_get
    let _ = c.get().ilog2();
    //~^ needless_nonzero_get
    let _ = d.get().is_power_of_two();
    //~^ needless_nonzero_get
    let _ = e.get().ilog10();
    //~^ needless_nonzero_get
}

fn signed_widths(f: NonZero<i8>, g: NonZero<i16>, h: NonZero<i64>, i: NonZero<i128>, j: NonZero<isize>) {
    let _ = f.get().is_negative();
    //~^ needless_nonzero_get
    let _ = g.get().is_positive();
    //~^ needless_nonzero_get
    let _ = h.get().leading_zeros();
    //~^ needless_nonzero_get
    let _ = i.get().trailing_zeros();
    //~^ needless_nonzero_get
    let _ = j.get().leading_zeros();
    //~^ needless_nonzero_get
}

fn aliases_and_receivers(a: NonZeroU32, b: NonZeroI32, r: &NonZero<u32>) {
    let _ = a.get().ilog2();
    //~^ needless_nonzero_get
    let _ = b.get().is_positive();
    //~^ needless_nonzero_get
    let _ = r.get().leading_zeros();
    //~^ needless_nonzero_get

    // more complex receiver expressions
    let _ = NonZero::new(5u32).unwrap().get().leading_zeros();
    //~^ needless_nonzero_get
    let _ = (a).get().trailing_zeros();
    //~^ needless_nonzero_get

    // multi-line chain
    let _ = a
        .get()
        //~^ needless_nonzero_get
        .leading_zeros();
}

fn operators(mut value: u32, other: u32, nz: NonZero<u32>) {
    let _ = other / nz.get();
    //~^ needless_nonzero_get
    let _ = other % nz.get();
    //~^ needless_nonzero_get
    value /= nz.get();
    //~^ needless_nonzero_get
    value %= nz.get();
    //~^ needless_nonzero_get
}

// The `NonZero` operator implementations carry `#[rustc_const_unstable(feature = "const_ops")]`,
// so they are not usable in a `const` context even though `Div`/`Rem` themselves are stable.
const fn const_operators(mut value: u32, nz: NonZero<u32>) -> u32 {
    value /= nz.get();
    value / nz.get()
}

// `NonZero::leading_zeros` is const-stable since 1.53.0, so the `get` can go even in a `const fn`.
const fn const_methods(nz: NonZero<u32>) -> u32 {
    nz.get().leading_zeros()
    //~^ needless_nonzero_get
}

fn no_lint(nz: NonZero<u32>, signed: NonZero<i32>, plain: u32) {
    // `NonZero`'s version returns `NonZero<u32>` rather than `u32`, so the `get` would only move
    // rather than disappear
    let _ = nz.get().bit_width();
    let _ = nz.get().count_ones();
    let _ = nz.get().isqrt();
    let _ = nz.get().checked_add(1);
    let _ = nz.get().saturating_mul(2);
    let _ = signed.get().abs();
    let _ = signed.get().cast_unsigned();
    let _ = signed.get().unsigned_abs();
    let _ = signed.get().wrapping_neg();
    let _ = signed.get().overflowing_neg();

    // `highest_one`/`lowest_one` return `Option<u32>` on integers but `u32` on `NonZero`
    let _ = nz.get().highest_one();
    let _ = nz.get().lowest_one();

    // no `NonZero` equivalent at all
    let _ = nz.get().to_string();
    let _ = nz.get().count_zeros();

    // `i32::ilog2` exists but `NonZero<i32>::ilog2` does not, so the impls must not cross over
    let _ = signed.get().ilog2();

    // The `NonZero` division and remainder implementations are unsigned-only.
    let signed_value = signed.get();
    let _ = signed_value / signed.get();
    let _ = signed_value % signed.get();

    // Primitive operators forward references, but the `NonZero` operators do not.
    let value_ref = &plain;
    let _ = value_ref / nz.get();
    let _ = value_ref % nz.get();

    let nz_ref = &nz;
    let _ = plain / nz_ref.get();
    let _ = plain % nz_ref.get();

    // not a `NonZero` receiver
    let _ = plain.leading_zeros();

    // `get` is not immediately followed by the method
    let x = nz.get();
    let _ = x.leading_zeros();

    // takes arguments
    let _ = nz.get().rotate_left(2);

    // a shadowing trait method must not be rewritten
    let _ = nz.get().shadowed();
}

trait Shadowed {
    fn shadowed(self) -> u32;
}

impl Shadowed for u32 {
    fn shadowed(self) -> u32 {
        self
    }
}

impl Shadowed for NonZero<u32> {
    fn shadowed(self) -> u32 {
        0
    }
}

// `NonZero::leading_zeros` was stabilized in 1.53.0
#[clippy::msrv = "1.52"]
fn below_msrv(nz: NonZero<u32>) {
    let _ = nz.get().leading_zeros();
}

#[clippy::msrv = "1.53"]
fn meets_msrv(nz: NonZero<u32>) {
    let _ = nz.get().leading_zeros();
    //~^ needless_nonzero_get
}

// `NonZero::ilog2` was stabilized in 1.67.0, later than `leading_zeros`
#[clippy::msrv = "1.66"]
fn below_ilog2_msrv(nz: NonZero<u32>) {
    let _ = nz.get().ilog2();
    let _ = nz.get().leading_zeros();
    //~^ needless_nonzero_get
}

// `Div<NonZero<_>>` and `Rem<NonZero<_>>` were stabilized in 1.51.0
#[clippy::msrv = "1.50"]
fn below_nonzero_div_msrv(value: u32, nz: NonZero<u32>) {
    let _ = value / nz.get();
    let _ = value % nz.get();
}

#[clippy::msrv = "1.51"]
fn meets_nonzero_div_msrv(value: u32, nz: NonZero<u32>) {
    let _ = value / nz.get();
    //~^ needless_nonzero_get
    let _ = value % nz.get();
    //~^ needless_nonzero_get
}

// `DivAssign<NonZero<_>>` and `RemAssign<NonZero<_>>` were stabilized in 1.79.0
#[clippy::msrv = "1.78"]
fn below_nonzero_div_assign_msrv(mut value: u32, nz: NonZero<u32>) {
    value /= nz.get();
    value %= nz.get();
}

#[clippy::msrv = "1.79"]
fn meets_nonzero_div_assign_msrv(mut value: u32, nz: NonZero<u32>) {
    value /= nz.get();
    //~^ needless_nonzero_get
    value %= nz.get();
    //~^ needless_nonzero_get
}

// The whole expression is written in the macro, so the suggestion would point at code the caller
// cannot edit.
macro_rules! leading_zeros_of_five {
    () => {{
        let nz = NonZero::new(5u32).unwrap();
        nz.get().leading_zeros()
    }};
}

// Only the receiver comes from the macro, so the removal span would start in the expansion and end
// at the call site.
macro_rules! five {
    () => {
        NonZero::new(5u32).unwrap()
    };
}

// Only `.get()` comes from the macro: the receiver is a macro argument and keeps its call site
// span.
macro_rules! get_of {
    ($nz:expr) => {
        $nz.get()
    };
}

fn from_macros(nz: NonZero<u32>) {
    let _ = leading_zeros_of_five!();
    let _ = five!().get().leading_zeros();
    let _ = get_of!(nz).leading_zeros();
    let _ = with_span!(span nz.get().leading_zeros());
}

fn main() {}
