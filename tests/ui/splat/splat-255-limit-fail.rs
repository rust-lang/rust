// ignore-tidy-file-linelength
//! Test `#[splat]` fails over the 255th argument index (or higher).
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
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is only supported on argument index 254 or less, this `#[splat]` is on index 255
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
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is only supported on argument index 254 or less, this `#[splat]` is on index 256
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
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is only supported on argument index 254 or less, this `#[splat]` is on index 255
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
    #[splat] (_a, _b): (u32, i8), //~ ERROR `#[splat]` is only supported on argument index 254 or less, this `#[splat]` is on index 256
    _: A,
) {}

// It's only the splatted index that's constrained to 255, not the argument count of the caller or callee.
fn more_than_255_splatted_args(#[splat] _t: Tuple256) {}

fn main() {
    let a = ();

    #[rustfmt::skip]
    more_than_255_splatted_args( //~ ERROR this splatted function takes 256 arguments, but 255 were provided [E0057]
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
        /* missing: a, */
    );

    #[rustfmt::skip]
    more_than_255_splatted_args( //~ ERROR this splatted function takes 256 arguments, but 257 were provided [E0057]
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
        a, /* unexpected: */ a,
    );

    #[rustfmt::skip]
    more_than_255_splatted_args( //~ ERROR this splatted function takes 256 arguments, but 512 were provided [E0057]
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
        /* unexpected: */
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
