// ignore-tidy-linelength
//! Test `#[splat]` fails over the 255th argument index (or higher).
//! FIXME(splat): The 255 argument limit is a temporary performance hack.

#![allow(incomplete_features)]
#![feature(splat)]
#![expect(dead_code)]

type A = ();

// These functions are deliberately formatted with 17 arguments in 15 lines, to show they have 255
// arguments.
#[rustfmt::skip]
fn s_255_terminal(
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is not supported on argument index 255
) {}

#[rustfmt::skip]
fn s_256_terminal(
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A,
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is not supported on argument index 256
) {}

#[rustfmt::skip]
fn s_255_non_terminal(
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is not supported on argument index 255
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
) {}

#[rustfmt::skip]
fn s_256_non_terminal(
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    _: A,
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is not supported on argument index 256
    _: A,
) {}

fn main() {}
