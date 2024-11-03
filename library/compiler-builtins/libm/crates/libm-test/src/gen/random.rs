//! A simple generator that produces deterministic random input, caching to use the same
//! inputs for all functions.

use std::sync::LazyLock;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::CachedInput;
use crate::{CheckCtx, GenerateInput};

const SEED: [u8; 32] = *b"3.141592653589793238462643383279";

/// Number of tests to run.
const NTESTS: usize = {
    if cfg!(optimizations_enabled) {
        if crate::emulated()
            || !cfg!(target_pointer_width = "64")
            || cfg!(all(target_arch = "x86_64", target_vendor = "apple"))
        {
            // Tests are pretty slow on non-64-bit targets, x86 MacOS, and targets that run
            // in QEMU.
            100_000
        } else {
            5_000_000
        }
    } else {
        // Without optimizations just run a quick check
        800
    }
};

/// Tested inputs.
static TEST_CASES: LazyLock<CachedInput> = LazyLock::new(|| make_test_cases(NTESTS));

/// The first argument to `jn` and `jnf` is the number of iterations. Make this a reasonable
/// value so tests don't run forever.
static TEST_CASES_JN: LazyLock<CachedInput> = LazyLock::new(|| {
    // Start with regular test cases
    let mut cases = (*TEST_CASES).clone();

    // These functions are extremely slow, limit them
    let ntests_jn = (NTESTS / 1000).max(80);
    cases.inputs_i32.truncate(ntests_jn);
    cases.inputs_f32.truncate(ntests_jn);
    cases.inputs_f64.truncate(ntests_jn);

    // It is easy to overflow the stack with these in debug mode
    let max_iterations = if cfg!(optimizations_enabled) && cfg!(target_pointer_width = "64") {
        0xffff
    } else if cfg!(windows) {
        0x00ff
    } else {
        0x0fff
    };

    let mut rng = ChaCha8Rng::from_seed(SEED);

    for case in cases.inputs_i32.iter_mut() {
        case.0 = rng.gen_range(3..=max_iterations);
    }

    cases
});

fn make_test_cases(ntests: usize) -> CachedInput {
    let mut rng = ChaCha8Rng::from_seed(SEED);

    // make sure we include some basic cases
    let mut inputs_i32 = vec![(0, 0, 0), (1, 1, 1), (-1, -1, -1)];
    let mut inputs_f32 = vec![
        (0.0, 0.0, 0.0),
        (f32::EPSILON, f32::EPSILON, f32::EPSILON),
        (f32::INFINITY, f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        (f32::MAX, f32::MAX, f32::MAX),
        (f32::MIN, f32::MIN, f32::MIN),
        (f32::MIN_POSITIVE, f32::MIN_POSITIVE, f32::MIN_POSITIVE),
        (f32::NAN, f32::NAN, f32::NAN),
    ];
    let mut inputs_f64 = vec![
        (0.0, 0.0, 0.0),
        (f64::EPSILON, f64::EPSILON, f64::EPSILON),
        (f64::INFINITY, f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        (f64::MAX, f64::MAX, f64::MAX),
        (f64::MIN, f64::MIN, f64::MIN),
        (f64::MIN_POSITIVE, f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        (f64::NAN, f64::NAN, f64::NAN),
    ];

    inputs_i32.extend((0..(ntests - inputs_i32.len())).map(|_| rng.gen::<(i32, i32, i32)>()));

    // Generate integers to get a full range of bitpatterns, then convert back to
    // floats.
    inputs_f32.extend((0..(ntests - inputs_f32.len())).map(|_| {
        let ints = rng.gen::<(u32, u32, u32)>();
        (f32::from_bits(ints.0), f32::from_bits(ints.1), f32::from_bits(ints.2))
    }));
    inputs_f64.extend((0..(ntests - inputs_f64.len())).map(|_| {
        let ints = rng.gen::<(u64, u64, u64)>();
        (f64::from_bits(ints.0), f64::from_bits(ints.1), f64::from_bits(ints.2))
    }));

    CachedInput { inputs_f32, inputs_f64, inputs_i32 }
}

/// Create a test case iterator.
pub fn get_test_cases<RustArgs>(ctx: &CheckCtx) -> impl Iterator<Item = RustArgs>
where
    CachedInput: GenerateInput<RustArgs>,
{
    let inputs = if ctx.fn_name_str == "jn" || ctx.fn_name_str == "jnf" {
        &TEST_CASES_JN
    } else {
        &TEST_CASES
    };
    inputs.get_cases()
}
