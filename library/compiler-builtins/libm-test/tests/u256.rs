//! Test the u256 implementation. the ops already get exercised reasonably well through the `f128`
//! routines, so this only does a few million fuzz iterations against GMP.

#![cfg(feature = "build-mpfr")]

use std::sync::LazyLock;

use libm::support::{HInt, i256, u256};
type BigInt = rug::Integer;

use libm_test::generate::random::SEED;
use libm_test::{MinInt, bigint_fuzz_iteration_count};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rug::Assign;
use rug::integer::Order;
use rug::ops::NotAssign;

static BIGINT_U256_MAX: LazyLock<BigInt> =
    LazyLock::new(|| BigInt::from_digits(&[u128::MAX, u128::MAX], Order::Lsf));

fn random_u256(rng: &mut ChaCha8Rng) -> u256 {
    let lo: u128 = rng.random();
    let hi: u128 = rng.random();
    u256 { lo, hi }
}

fn random_i256(rng: &mut ChaCha8Rng) -> i256 {
    random_u256(rng).signed()
}

fn assign_bigint_u256(bx: &mut BigInt, x: u256) {
    bx.assign(x.hi);
    *bx <<= 128;
    *bx += x.lo;
}

fn assign_bigint_i256(bx: &mut BigInt, x: i256) {
    bx.assign(x.hi);
    *bx <<= 128;
    *bx += x.lo;
}

/// Note that this destroys the result in `bx`.
fn from_bigint_u256(bx: &mut BigInt) -> u256 {
    // Truncate so the result fits into `[u128; 2]`. This makes all ops overflowing.
    *bx &= &*BIGINT_U256_MAX;
    let mut bres = [0u128, 0];
    bx.write_digits(&mut bres, Order::Lsf);
    bx.assign(0); // prevent accidental reuse
    u256 {
        lo: bres[0],
        hi: bres[1],
    }
}

/// Note that this destroys the result in `bx`.
fn from_bigint_i256(bx: &mut BigInt) -> i256 {
    // Truncate so the result fits into `[u128; 2]`. This makes all ops overflowing.
    *bx &= &*BIGINT_U256_MAX;
    let lo = bx.to_u128_wrapping();
    *bx >>= 128;
    let hi = bx.to_i128_wrapping();
    bx.assign(0); // prevent accidental reuse
    i256 { hi, lo }
}

#[track_caller]
fn assert_same_u256(msg: impl Fn() -> String, actual: u256, expected_big: &mut BigInt) {
    let expected = from_bigint_u256(expected_big);
    if actual != expected {
        let mut act_big = BigInt::new();
        assign_bigint_u256(&mut act_big, actual);
        panic!(
            "Test failure: {}\n\
            actual:   {act_big}\n\
            expected: {expected_big}\n\
            actual:   {actual:#x}\n\
            expected: {expected:#x}\
            ",
            msg()
        )
    }
}

#[track_caller]
fn assert_same_i256(msg: impl Fn() -> String, actual: i256, expected_big: &mut BigInt) {
    let expected = from_bigint_i256(expected_big);
    if actual != expected {
        let mut act_big = BigInt::new();
        assign_bigint_i256(&mut act_big, actual);
        panic!(
            "Test failure: {}\n\
            actual:   {act_big}\n\
            expected: {expected_big}\n\
            actual:   {actual:#x}\n\
            expected: {expected:#x}\
            ",
            msg()
        )
    }
}

/// Verify the test setup.
#[test]
fn mp_u256_roundtrip() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        assign_bigint_u256(&mut bx, x);
        assert_eq!(from_bigint_u256(&mut bx), x);
    }

    // Check wraparound
    assign_bigint_u256(&mut bx, u256::MAX);
    bx += 1;
    assert_eq!(from_bigint_u256(&mut bx), u256::MIN);
    assign_bigint_u256(&mut bx, u256::MIN);
    bx -= 1;
    assert_eq!(from_bigint_u256(&mut bx), u256::MAX);
}

/// Verify the test setup.
#[test]
fn mp_i256_roundtrip() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        assert_eq!(from_bigint_i256(&mut bx), x);
    }

    // Check wraparound
    assign_bigint_i256(&mut bx, i256::MAX);
    bx += 1;
    assert_eq!(from_bigint_i256(&mut bx), i256::MIN);
    assign_bigint_i256(&mut bx, i256::MIN);
    bx -= 1;
    assert_eq!(from_bigint_i256(&mut bx), i256::MAX);
}

#[test]
fn mp_u256_ord() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let y = random_u256(&mut rng);
        assign_bigint_u256(&mut bx, x);
        assign_bigint_u256(&mut by, y);

        assert_eq!(x.cmp(&y), bx.cmp(&by), "cmp({x:#x}, {y:#x})");
    }
}

#[test]
fn mp_i256_ord() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let y = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        assign_bigint_i256(&mut by, y);

        assert_eq!(x.cmp(&y), bx.cmp(&by), "cmp({x:#x}, {y:#x})");
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
        assign_bigint_u256(&mut bx, x);
        assign_bigint_u256(&mut by, y);
        let actual = x | y;
        bx |= &by;
        assert_same_u256(|| format!("{x:#x} ^ {y:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_bitor() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let y = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        assign_bigint_i256(&mut by, y);
        let actual = x | y;
        bx |= &by;
        assert_same_i256(|| format!("{x:#x} ^ {y:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_u256_not() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        assign_bigint_u256(&mut bx, x);
        let actual = !x;
        bx.not_assign();
        assert_same_u256(|| format!("!{x:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_not() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        let actual = !x;
        bx.not_assign();
        assert_same_i256(|| format!("!{x:#x}"), actual, &mut bx);
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
        assign_bigint_u256(&mut bx, x);
        assign_bigint_u256(&mut by, y);
        // Emulate wrapping semantics with panicking ops
        let actual = if u256::MAX - x >= y {
            x + y
        } else {
            // otherwise (u256::MAX - x) < y, so the wrapped result is
            // (x + y) - (u256::MAX + 1) == y - (u256::MAX - x) - 1
            y - (u256::MAX - x) - 1_u128.widen()
        };
        bx += &by;
        assert_same_u256(|| format!("{x:#x} + {y:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_add() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let y = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        assign_bigint_i256(&mut by, y);

        // Emulate wrapping semantics with panicking ops
        let actual = if x > i256::ZERO && y > i256::MAX - x {
            // Overflow condition
            (x + i256::MIN) + (y + i256::MIN)
        } else if x < i256::ZERO && y < i256::MIN - x {
            // Underflow condition
            (x - i256::MIN) + (y - i256::MIN)
        } else {
            // Otherwise there is no overflow
            x + y
        };
        bx += &by;
        assert_same_i256(|| format!("{x:#x} + {y:#x}"), actual, &mut bx);
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
        assign_bigint_u256(&mut bx, x);
        assign_bigint_u256(&mut by, y);

        // since the operators (may) panic on overflow,
        // we should test something that doesn't
        let actual = if x >= y { x - y } else { y - x };
        bx -= &by;
        bx.abs_mut();
        assert_same_u256(|| format!("{x:#x} - {y:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_sub() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();
    let mut by = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let y = random_i256(&mut rng);
        assign_bigint_i256(&mut bx, x);
        assign_bigint_i256(&mut by, y);
        dbg!(&bx, &by);

        // Emulate wrapping semantics with panicking ops
        let actual = if y > i256::ZERO && x < i256::MIN + y {
            (x - i256::MIN) - (y + i256::MIN)
        } else if y < i256::ZERO && x > i256::MAX + y {
            (x + i256::MIN) - (y - i256::MIN)
        } else {
            x - y
        };
        bx -= &by;
        assert_same_i256(|| format!("{x:#x} - {y:#x}"), actual, &mut bx);
    }
}

#[test]
fn mp_u256_shl() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint_u256(&mut bx, x);
        let actual = x << shift;
        bx <<= shift;
        assert_same_u256(|| format!("{x:#x} << {shift}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_shl() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint_i256(&mut bx, x);
        let actual = x << shift;
        bx <<= shift;
        assert_same_i256(|| format!("{x:#x} << {shift}"), actual, &mut bx);
    }
}

#[test]
fn mp_u256_shr() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_u256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint_u256(&mut bx, x);
        let actual = x >> shift;
        bx >>= shift;
        assert_same_u256(|| format!("{x:#x} >> {shift}"), actual, &mut bx);
    }
}

#[test]
fn mp_i256_shr() {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    let mut bx = BigInt::new();

    for _ in 0..bigint_fuzz_iteration_count() {
        let x = random_i256(&mut rng);
        let shift: u32 = rng.random_range(0..256);
        assign_bigint_i256(&mut bx, x);
        let actual = x >> shift;
        bx >>= shift;
        assert_same_i256(|| format!("{x:#x} >> {shift}"), actual, &mut bx);
    }
}

#[test]
fn mp_u256_u128_widen_mul() {
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
        assert_same_u256(
            || format!("{x:#034x}.widen_mul({y:#034x})"),
            actual,
            &mut bx,
        );
    }
}
