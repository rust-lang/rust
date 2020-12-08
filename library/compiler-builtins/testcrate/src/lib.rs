//! This crate is for integration testing and fuzz testing of functions in `compiler-builtins`. This
//! includes publicly documented intrinsics and some internal alternative implementation functions
//! such as `usize_leading_zeros_riscv` (which are tested because they are configured for
//! architectures not tested by the CI).
//!
//! The general idea is to use a combination of edge case testing and randomized fuzz testing. The
//! edge case testing is crucial for checking cases like where both inputs are equal or equal to
//! special values such as `i128::MIN`, which is unlikely for the random fuzzer by itself to
//! encounter. The randomized fuzz testing is specially designed to cover wide swaths of search
//! space in as few iterations as possible. See `fuzz_values` in `testcrate/tests/misc.rs` for an
//! example.
//!
//! Some floating point tests are disabled for specific architectures, because they do not have
//! correct rounding.
#![no_std]

use compiler_builtins::float::Float;
use compiler_builtins::int::Int;

use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro128StarStar;

/// Sets the number of fuzz iterations run for most tests. In practice, the vast majority of bugs
/// are caught by the edge case testers. Most of the remaining bugs triggered by more complex
/// sequences are caught well within 10_000 fuzz iterations. For classes of algorithms like division
/// that are vulnerable to rare edge cases, we want 1_000_000 iterations to be more confident. In
/// practical CI, however, we only want to run the more strenuous test once to catch algorithmic
/// level bugs, and run the 10_000 iteration test on most targets. Target-dependent bugs are likely
/// to involve miscompilation and misconfiguration that is likely to break algorithms in quickly
/// caught ways. We choose to configure `N = 1_000_000` iterations for `x86_64` targets (and if
/// debug assertions are disabled. Tests without `--release` would take too long) which are likely
/// to have fast hardware, and run `N = 10_000` for all other targets.
pub const N: u32 = if cfg!(target_arch = "x86_64") && !cfg!(debug_assertions) {
    1_000_000
} else {
    10_000
};

/// Random fuzzing step. When run several times, it results in excellent fuzzing entropy such as:
/// 11110101010101011110111110011111
/// 10110101010100001011101011001010
/// 1000000000000000
/// 10000000000000110111110000001010
/// 1111011111111101010101111110101
/// 101111111110100000000101000000
/// 10000000110100000000100010101
/// 1010101010101000
fn fuzz_step<I: Int>(rng: &mut Xoshiro128StarStar, x: &mut I) {
    let ones = !I::ZERO;
    let bit_indexing_mask: u32 = I::BITS - 1;
    // It happens that all the RNG we need can come from one call. 7 bits are needed to index a
    // worst case 128 bit integer, and there are 4 indexes that need to be made plus 4 bits for
    // selecting operations
    let rng32 = rng.next_u32();

    // Randomly OR, AND, and XOR randomly sized and shifted continuous strings of
    // ones with `lhs` and `rhs`.
    let r0 = bit_indexing_mask & rng32;
    let r1 = bit_indexing_mask & (rng32 >> 7);
    let mask = ones.wrapping_shl(r0).rotate_left(r1);
    match (rng32 >> 14) % 4 {
        0 => *x |= mask,
        1 => *x &= mask,
        // both 2 and 3 to make XORs as common as ORs and ANDs combined
        _ => *x ^= mask,
    }

    // Alternating ones and zeros (e.x. 0b1010101010101010). This catches second-order
    // problems that might occur for algorithms with two modes of operation (potentially
    // there is some invariant that can be broken and maintained via alternating between modes,
    // breaking the algorithm when it reaches the end).
    let mut alt_ones = I::ONE;
    for _ in 0..(I::BITS / 2) {
        alt_ones <<= 2;
        alt_ones |= I::ONE;
    }
    let r0 = bit_indexing_mask & (rng32 >> 16);
    let r1 = bit_indexing_mask & (rng32 >> 23);
    let mask = alt_ones.wrapping_shl(r0).rotate_left(r1);
    match rng32 >> 30 {
        0 => *x |= mask,
        1 => *x &= mask,
        _ => *x ^= mask,
    }
}

// We need macros like this, because `#![no_std]` prevents us from using iterators
macro_rules! edge_cases {
    ($I:ident, $case:ident, $inner:block) => {
        for i0 in 0..$I::FUZZ_NUM {
            let mask_lo = (!$I::UnsignedInt::ZERO).wrapping_shr($I::FUZZ_LENGTHS[i0] as u32);
            for i1 in i0..I::FUZZ_NUM {
                let mask_hi =
                    (!$I::UnsignedInt::ZERO).wrapping_shl($I::FUZZ_LENGTHS[i1 - i0] as u32);
                let $case = I::from_unsigned(mask_lo & mask_hi);
                $inner
            }
        }
    };
}

/// Feeds a series of fuzzing inputs to `f`. The fuzzer first uses an algorithm designed to find
/// edge cases, followed by a more random fuzzer that runs `n` times.
pub fn fuzz<I: Int, F: FnMut(I)>(n: u32, mut f: F) {
    // edge case tester. Calls `f` 210 times for u128.
    // zero gets skipped by the loop
    f(I::ZERO);
    edge_cases!(I, case, {
        f(case);
    });

    // random fuzzer
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x: I = Int::ZERO;
    for _ in 0..n {
        fuzz_step(&mut rng, &mut x);
        f(x)
    }
}

/// The same as `fuzz`, except `f` has two inputs.
pub fn fuzz_2<I: Int, F: Fn(I, I)>(n: u32, f: F) {
    // Check cases where the first and second inputs are zero. Both call `f` 210 times for `u128`.
    edge_cases!(I, case, {
        f(I::ZERO, case);
    });
    edge_cases!(I, case, {
        f(case, I::ZERO);
    });
    // Nested edge tester. Calls `f` 44100 times for `u128`.
    edge_cases!(I, case0, {
        edge_cases!(I, case1, {
            f(case0, case1);
        })
    });

    // random fuzzer
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x: I = I::ZERO;
    let mut y: I = I::ZERO;
    for _ in 0..n {
        fuzz_step(&mut rng, &mut x);
        fuzz_step(&mut rng, &mut y);
        f(x, y)
    }
}

/// Tester for shift functions
pub fn fuzz_shift<I: Int, F: Fn(I, u32)>(f: F) {
    // Shift functions are very simple and do not need anything other than shifting a small
    // set of random patterns for every fuzz length.
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x: I = Int::ZERO;
    for i in 0..I::FUZZ_NUM {
        fuzz_step(&mut rng, &mut x);
        f(x, Int::ZERO);
        f(x, I::FUZZ_LENGTHS[i] as u32);
    }
}

fn fuzz_float_step<F: Float>(rng: &mut Xoshiro128StarStar, f: &mut F) {
    let rng32 = rng.next_u32();
    // we need to fuzz the different parts of the float separately, because the masking on larger
    // significands will tend to set the exponent to all ones or all zeros frequently

    // sign bit fuzzing
    let sign = (rng32 & 1) != 0;

    // exponent fuzzing. Only 4 bits for the selector needed.
    let ones = (F::Int::ONE << F::EXPONENT_BITS) - F::Int::ONE;
    let r0 = (rng32 >> 1) % F::EXPONENT_BITS;
    let r1 = (rng32 >> 5) % F::EXPONENT_BITS;
    // custom rotate shift. Note that `F::Int` is unsigned, so we can shift right without smearing
    // the sign bit.
    let mask = if r1 == 0 {
        ones.wrapping_shr(r0)
    } else {
        let tmp = ones.wrapping_shr(r0);
        (tmp.wrapping_shl(r1) | tmp.wrapping_shr(F::EXPONENT_BITS - r1)) & ones
    };
    let mut exp = (f.repr() & F::EXPONENT_MASK) >> F::SIGNIFICAND_BITS;
    match (rng32 >> 9) % 4 {
        0 => exp |= mask,
        1 => exp &= mask,
        _ => exp ^= mask,
    }

    // significand fuzzing
    let mut sig = f.repr() & F::SIGNIFICAND_MASK;
    fuzz_step(rng, &mut sig);
    sig &= F::SIGNIFICAND_MASK;

    *f = F::from_parts(sign, exp, sig);
}

macro_rules! float_edge_cases {
    ($F:ident, $case:ident, $inner:block) => {
        for exponent in [
            F::Int::ZERO,
            F::Int::ONE,
            F::Int::ONE << (F::EXPONENT_BITS / 2),
            (F::Int::ONE << (F::EXPONENT_BITS - 1)) - F::Int::ONE,
            F::Int::ONE << (F::EXPONENT_BITS - 1),
            (F::Int::ONE << (F::EXPONENT_BITS - 1)) + F::Int::ONE,
            (F::Int::ONE << F::EXPONENT_BITS) - F::Int::ONE,
        ]
        .iter()
        {
            for significand in [
                F::Int::ZERO,
                F::Int::ONE,
                F::Int::ONE << (F::SIGNIFICAND_BITS / 2),
                (F::Int::ONE << (F::SIGNIFICAND_BITS - 1)) - F::Int::ONE,
                F::Int::ONE << (F::SIGNIFICAND_BITS - 1),
                (F::Int::ONE << (F::SIGNIFICAND_BITS - 1)) + F::Int::ONE,
                (F::Int::ONE << F::SIGNIFICAND_BITS) - F::Int::ONE,
            ]
            .iter()
            {
                for sign in [false, true].iter() {
                    let $case = F::from_parts(*sign, *exponent, *significand);
                    $inner
                }
            }
        }
    };
}

pub fn fuzz_float<F: Float, E: Fn(F)>(n: u32, f: E) {
    float_edge_cases!(F, case, {
        f(case);
    });

    // random fuzzer
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x = F::ZERO;
    for _ in 0..n {
        fuzz_float_step(&mut rng, &mut x);
        f(x);
    }
}

pub fn fuzz_float_2<F: Float, E: Fn(F, F)>(n: u32, f: E) {
    float_edge_cases!(F, case0, {
        float_edge_cases!(F, case1, {
            f(case0, case1);
        });
    });

    // random fuzzer
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x = F::ZERO;
    let mut y = F::ZERO;
    for _ in 0..n {
        fuzz_float_step(&mut rng, &mut x);
        fuzz_float_step(&mut rng, &mut y);
        f(x, y)
    }
}
