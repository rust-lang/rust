//! This crate is for integration testing and fuzz testing of functions in `compiler-builtins`. This
//! includes publicly documented intrinsics and some internal alternative implementation functions
//! such as `usize_leading_zeros_riscv` (which are tested because they are configured for
//! architectures not tested by the CI).
//!
//! The general idea is to use a combination of edge case testing and randomized fuzz testing. The
//! edge case testing is crucial for checking cases like where both inputs are equal or equal to
//! special values such as `i128::MIN`, which is unlikely for the random fuzzer by itself to
//! encounter. The randomized fuzz testing is specially designed to cover wide swaths of search
//! space in as few iterations as possible. See `fuzz_values` in `builtins-test/tests/misc.rs` for
//! an example.
//!
//! Some floating point tests are disabled for specific architectures, because they do not have
//! correct rounding.
#![no_std]
#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]

pub mod bench;
extern crate alloc;

use compiler_builtins::float::Float;
use compiler_builtins::int::{Int, MinInt};
use rand_xoshiro::Xoshiro128StarStar;
use rand_xoshiro::rand_core::{RngCore, SeedableRng};

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

/// Additional constants that determine how the integer gets fuzzed.
trait FuzzInt: MinInt {
    /// LUT used for maximizing the space covered and minimizing the computational cost of fuzzing
    /// in `builtins-test`. For example, Self = u128 produces [0,1,2,7,8,15,16,31,32,63,64,95,96,
    /// 111,112,119,120,125,126,127].
    const FUZZ_LENGTHS: [u8; 20] = make_fuzz_lengths(Self::BITS);

    /// The number of entries of `FUZZ_LENGTHS` actually used. The maximum is 20 for u128.
    const FUZZ_NUM: usize = {
        let log2 = Self::BITS.ilog2() as usize;
        if log2 == 3 {
            // case for u8
            6
        } else {
            // 3 entries on each extreme, 2 in the middle, and 4 for each scale of intermediate
            // boundaries.
            8 + (4 * (log2 - 4))
        }
    };
}

impl<I> FuzzInt for I where I: MinInt {}

const fn make_fuzz_lengths(bits: u32) -> [u8; 20] {
    let mut v = [0u8; 20];
    v[0] = 0;
    v[1] = 1;
    v[2] = 2; // important for parity and the iX::MIN case when reversed
    let mut i = 3;

    // No need for any more until the byte boundary, because there should be no algorithms
    // that are sensitive to anything not next to byte boundaries after 2. We also scale
    // in powers of two, which is important to prevent u128 corner tests from getting too
    // big.
    let mut l = 8;
    loop {
        if l >= ((bits / 2) as u8) {
            break;
        }
        // get both sides of the byte boundary
        v[i] = l - 1;
        i += 1;
        v[i] = l;
        i += 1;
        l *= 2;
    }

    if bits != 8 {
        // add the lower side of the middle boundary
        v[i] = ((bits / 2) - 1) as u8;
        i += 1;
    }

    // We do not want to jump directly from the Self::BITS/2 boundary to the Self::BITS
    // boundary because of algorithms that split the high part up. We reverse the scaling
    // as we go to Self::BITS.
    let mid = i;
    let mut j = 1;
    loop {
        v[i] = (bits as u8) - (v[mid - j]) - 1;
        if j == mid {
            break;
        }
        i += 1;
        j += 1;
    }
    v
}

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
            let mask_lo = (!$I::Unsigned::ZERO).wrapping_shr($I::FUZZ_LENGTHS[i0] as u32);
            for i1 in i0..I::FUZZ_NUM {
                let mask_hi = (!$I::Unsigned::ZERO).wrapping_shl($I::FUZZ_LENGTHS[i1 - i0] as u32);
                let $case = I::from_unsigned(mask_lo & mask_hi);
                $inner
            }
        }
    };
}

/// Feeds a series of fuzzing inputs to `f`. The fuzzer first uses an algorithm designed to find
/// edge cases, followed by a more random fuzzer that runs `n` times.
pub fn fuzz<I: Int, F: FnMut(I)>(n: u32, mut f: F)
where
    <I as MinInt>::Unsigned: Int,
{
    // edge case tester. Calls `f` 210 times for u128.
    // zero gets skipped by the loop
    f(I::ZERO);
    edge_cases!(I, case, {
        f(case);
    });

    // random fuzzer
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x: I = MinInt::ZERO;
    for _ in 0..n {
        fuzz_step(&mut rng, &mut x);
        f(x)
    }
}

/// The same as `fuzz`, except `f` has two inputs.
pub fn fuzz_2<I: Int, F: Fn(I, I)>(n: u32, f: F)
where
    <I as MinInt>::Unsigned: Int,
{
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
    let mut x: I = MinInt::ZERO;
    for i in 0..I::FUZZ_NUM {
        fuzz_step(&mut rng, &mut x);
        f(x, MinInt::ZERO);
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
    let ones = (F::Int::ONE << F::EXP_BITS) - F::Int::ONE;
    let r0 = (rng32 >> 1) % F::EXP_BITS;
    let r1 = (rng32 >> 5) % F::EXP_BITS;
    // custom rotate shift. Note that `F::Int` is unsigned, so we can shift right without smearing
    // the sign bit.
    let mask = if r1 == 0 {
        ones.wrapping_shr(r0)
    } else {
        let tmp = ones.wrapping_shr(r0);
        (tmp.wrapping_shl(r1) | tmp.wrapping_shr(F::EXP_BITS - r1)) & ones
    };
    let mut exp = (f.to_bits() & F::EXP_MASK) >> F::SIG_BITS;
    match (rng32 >> 9) % 4 {
        0 => exp |= mask,
        1 => exp &= mask,
        _ => exp ^= mask,
    }

    // significand fuzzing
    let mut sig = f.to_bits() & F::SIG_MASK;
    fuzz_step(rng, &mut sig);
    sig &= F::SIG_MASK;

    *f = F::from_parts(sign, exp, sig);
}

macro_rules! float_edge_cases {
    ($F:ident, $case:ident, $inner:block) => {
        for exponent in [
            F::Int::ZERO,
            F::Int::ONE,
            F::Int::ONE << (F::EXP_BITS / 2),
            (F::Int::ONE << (F::EXP_BITS - 1)) - F::Int::ONE,
            F::Int::ONE << (F::EXP_BITS - 1),
            (F::Int::ONE << (F::EXP_BITS - 1)) + F::Int::ONE,
            (F::Int::ONE << F::EXP_BITS) - F::Int::ONE,
        ]
        .iter()
        {
            for significand in [
                F::Int::ZERO,
                F::Int::ONE,
                F::Int::ONE << (F::SIG_BITS / 2),
                (F::Int::ONE << (F::SIG_BITS - 1)) - F::Int::ONE,
                F::Int::ONE << (F::SIG_BITS - 1),
                (F::Int::ONE << (F::SIG_BITS - 1)) + F::Int::ONE,
                (F::Int::ONE << F::SIG_BITS) - F::Int::ONE,
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

/// Perform an operation using builtin types if available, falling back to apfloat if not.
#[macro_export]
macro_rules! apfloat_fallback {
    (
        $float_ty:ty,
        // Type name in `rustc_apfloat::ieee`. Not a full path, it automatically gets the prefix.
        $apfloat_ty:ident,
        // Cfg expression for when builtin system operations should be used
        $sys_available:meta,
        // The expression to run. This expression may use `FloatTy` for its signature.
        // Optionally, the final conversion back to a float can be suppressed using
        // `=> no_convert` (for e.g. operations that return a bool).
        //
        // If the apfloat needs a different operation, it can be provided here.
        $op:expr $(=> $convert:ident)? $(; $apfloat_op:expr)?,
        // Arguments that get passed to `$op` after converting to a float
        $($arg:expr),+
        $(,)?
    ) => {{
        #[cfg($sys_available)]
        let ret = {
            type FloatTy = $float_ty;
            $op( $($arg),+ )
        };

        #[cfg(not($sys_available))]
        let ret = {
            use rustc_apfloat::Float;
            type FloatTy = rustc_apfloat::ieee::$apfloat_ty;

            apfloat_fallback!(@inner
                fty: $float_ty,
                // Apply a conversion to `FloatTy` to each arg, then pass all args to `$op`
                op_res: $op( $(FloatTy::from_bits($arg.to_bits().into())),+ ),
                $(apfloat_op: $apfloat_op, )?
                $(conv_opts: $convert,)?
                args: $($arg),+
            )
        };

        ret
    }};

    // Operations that do not need converting back to a float
    (@inner fty: $float_ty:ty, op_res: $val:expr, conv_opts: no_convert, args: $($_arg:expr),+) => {
        $val
    };

    // Some apfloat operations return a `StatusAnd` that we need to extract the value from. This
    // is the default.
    (@inner fty: $float_ty:ty, op_res: $val:expr, args: $($_arg:expr),+) => {{
        // ignore the status, just get the value
        let unwrapped = $val.value;

        <$float_ty>::from_bits(FloatTy::to_bits(unwrapped).try_into().unwrap())
    }};

    // This is the case where we can't use the same expression for the default builtin and
    // nonstandard apfloat fallback (e.g. `as` casts in std are normal functions in apfloat, so
    // two separate expressions must be specified.
    (@inner
        fty: $float_ty:ty, op_res: $_val:expr,
        apfloat_op: $apfloat_op:expr, args: $($arg:expr),+
    ) => {{
        $apfloat_op($($arg),+)
    }};
}
