//@ run-pass
// ignore-tidy-linelength
//! Test `#[splat]` on the 255th argument index (or lower).
//! FIXME(splat): The 255 argument limit is a temporary performance hack.

#![allow(incomplete_features)]
#![feature(splat)]
#![expect(dead_code)]

type A = ();

// These types and functions are deliberately formatted with 17 arguments in 15 lines, to show they
// have ~255 arguments.
#[rustfmt::skip]
type Tuple256 = (
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A,
    A,
);

#[rustfmt::skip]
fn s_253_terminal(
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
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    #[splat] (_a, _b): (u32, i8),
) {}

#[rustfmt::skip]
fn s_254_terminal(
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
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    #[splat] (_a, _b): (u32, i8),
) {}

#[rustfmt::skip]
fn s_254_non_terminal_272_args(
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
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
    #[splat] (_a, _b): (u32, i8),
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
) {}

#[rustfmt::skip]
fn s_0_initial_253_args(
    #[splat] (_a, _b): (u32, i8),
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
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
) {}

#[rustfmt::skip]
fn s_0_initial_254_args(
    #[splat] (_a, _b): (u32, i8),
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
    _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A, _: A,
) {}

#[rustfmt::skip]
fn s_0_initial_255_args(
    #[splat] (_a, _b): (u32, i8),
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
) {}

// It's only the splatted index that's constrained to 255, not the argument count of the caller or callee.
#[rustfmt::skip]
fn s_0_initial_256_args(
    #[splat] (_a, _b): (u32, i8),
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
) {}

fn more_than_255_splatted_args(#[splat] _t: Tuple256) {}

fn main() {
    let a = ();

    #[rustfmt::skip]
    more_than_255_splatted_args(
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a,
    );
}
