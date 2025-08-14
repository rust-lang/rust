//! Test the u256 implementation. the ops already get exercised reasonably well through the `f128`
//! routines, so this only does a few million fuzz iterations against GMP.

#![cfg(feature = "build-mpfr")]

use std::sync::LazyLock;

use libm::support::{HInt, u256};
type BigInt = rug::Integer;

use libm_test::bigint_fuzz_iteration_count;
use libm_test::generate::random::SEED;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rug::Assign;
use rug::integer::Order;
use rug::ops::NotAssign;

static BIGINT_U256_MAX: LazyLock<BigInt> =
    LazyLock::new(|| BigInt::from_digits(&[u128::MAX, u128::MAX], Order::Lsf));

/// Copied from the test module.
fn hexu(v: u256) -> String {
    format!("0x{:032x}{:032x}", v.hi, v.lo)
}

fn random_u256(rng: &mut ChaCha8Rng) -> u256 {
    let lo: u128 = rng.random();
    let hi: u128 = rng.random();
    u256 { lo, hi }
}

fn assign_bigint(bx: &mut BigInt, x: u256) {
    bx.assign_digits(&[x.lo, x.hi], Order::Lsf);
}

fn from_bigint(bx: &mut BigInt) -> u256 {
    // Truncate so the result fits into `[u128; 2]`. This makes all ops overflowing.
    *bx &= &*BIGINT_U256_MAX;
    let mut bres = [0u128, 0];
    bx.write_digits(&mut bres, Order::Lsf);
    bx.assign(0);
    u256 {
        lo: bres[0],
        hi: bres[1],
    }
}

fn check_one(
    x: impl FnOnce() -> String,
    y: impl FnOnce() -> Option<String>,
    actual: u256,
    expected: &mut BigInt,
) {
    let expected = from_bigint(expected);
    if actual != expected {
        let xmsg = x();
        let ymsg = y().map(|y| format!("y:        {y}\n")).unwrap_or_default();
        panic!(
            "Results do not match\n\
            input:    {xmsg}\n\
            {ymsg}\
            actual:   {}\n\
            expected: {}\
            ",
            hexu(actual),
            hexu(expected),
        )
    }
}

#[test]
fn mp_u256_bitor() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let y = random_u256(&mut rng);
        assign_bigint(&mut bx, x);
        assign_bigint(&mut by, y);
        let actual = x | y;
        bx |= &by;
        check_one(|| hexu(x), || Some(hexu(y)), actual, &mut bx);
    }
}

#[test]
fn mp_u256_not() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        assign_bigint(&mut bx, x);
        let actual = !x;
        bx.not_assign();
        check_one(|| hexu(x), || None, actual, &mut bx);
    }
}

#[test]
fn mp_u256_add() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let y = random_u256(&mut rng);
        assign_bigint(&mut bx, x);
        assign_bigint(&mut by, y);
        let actual = if u256::MAX - x >= y {
            x + y
        } else {
            // otherwise (u256::MAX - x) < y, so the wrapped result is
            // (x + y) - (u256::MAX + 1) == y - (u256::MAX - x) - 1
            y - (u256::MAX - x) - 1_u128.widen()
        };
        bx += &by;
        check_one(|| hexu(x), || Some(hexu(y)), actual, &mut bx);
    }
}

#[test]
fn mp_u256_sub() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let y = random_u256(&mut rng);
        assign_bigint(&mut bx, x);
        assign_bigint(&mut by, y);

        // since the operators (may) panic on overflow,
        // we should test something that doesn't
        let actual = if x >= y { x - y } else { y - x };
        bx -= &by;
        bx.abs_mut();
        check_one(|| hexu(x), || Some(hexu(y)), actual, &mut bx);
    }
}

#[test]
fn mp_u256_shl() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint(&mut bx, x);
        let actual = x << shift;
        bx <<= shift;
        check_one(|| hexu(x), || Some(shift.to_string()), actual, &mut bx);
    }
}

#[test]
fn mp_u256_shr() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint(&mut bx, x);
        let actual = x >> shift;
        bx >>= shift;
        check_one(|| hexu(x), || Some(shift.to_string()), actual, &mut bx);
    }
}

#[test]
fn mp_u256_widen_mul() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x: u128 = rng.random();
        let y: u128 = rng.random();
        bx.assign(x);
        by.assign(y);
        let actual = x.widen_mul(y);
        bx *= &by;
        check_one(
            || format!("{x:#034x}"),
            || Some(format!("{y:#034x}")),
            actual,
            &mut bx,
        );
    }
}
