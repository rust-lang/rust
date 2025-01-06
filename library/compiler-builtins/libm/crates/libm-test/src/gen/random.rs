use std::env;
use std::ops::RangeInclusive;
use std::sync::LazyLock;

use libm::support::Float;
use rand::distributions::{Alphanumeric, Standard};
use rand::prelude::Distribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::KnownSize;
use crate::run_cfg::{int_range, iteration_count};
use crate::{CheckCtx, GeneratorKind};

pub(crate) const SEED_ENV: &str = "LIBM_SEED";

pub(crate) static SEED: LazyLock<[u8; 32]> = LazyLock::new(|| {
    let s = env::var(SEED_ENV).unwrap_or_else(|_| {
        let mut rng = rand::thread_rng();
        (0..32).map(|_| rng.sample(Alphanumeric) as char).collect()
    });

    s.as_bytes().try_into().unwrap_or_else(|_| {
        panic!("Seed must be 32 characters, got `{s}`");
    })
});

/// Generate a sequence of random values of this type.
pub trait RandomInput {
    fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self>;
}

/// Generate a sequence of deterministically random floats.
fn random_floats<F: Float>(count: u64) -> impl Iterator<Item = F>
where
    Standard: Distribution<F::Int>,
{
    let mut rng = ChaCha8Rng::from_seed(*SEED);

    // Generate integers to get a full range of bitpatterns (including NaNs), then convert back
    // to the float type.
    (0..count).map(move |_| F::from_bits(rng.gen::<F::Int>()))
}

/// Generate a sequence of deterministically random `i32`s within a specified range.
fn random_ints(count: u64, range: RangeInclusive<i32>) -> impl Iterator<Item = i32> {
    let mut rng = ChaCha8Rng::from_seed(*SEED);
    (0..count).map(move |_| rng.gen_range::<i32, _>(range.clone()))
}

macro_rules! impl_random_input {
    ($fty:ty) => {
        impl RandomInput for ($fty,) {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let count = iteration_count(ctx, GeneratorKind::Random, 0);
                let iter = random_floats(count).map(|f: $fty| (f,));
                KnownSize::new(iter, count)
            }
        }

        impl RandomInput for ($fty, $fty) {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let count0 = iteration_count(ctx, GeneratorKind::Random, 0);
                let count1 = iteration_count(ctx, GeneratorKind::Random, 1);
                let iter = random_floats(count0)
                    .flat_map(move |f1: $fty| random_floats(count1).map(move |f2: $fty| (f1, f2)));
                KnownSize::new(iter, count0 * count1)
            }
        }

        impl RandomInput for ($fty, $fty, $fty) {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let count0 = iteration_count(ctx, GeneratorKind::Random, 0);
                let count1 = iteration_count(ctx, GeneratorKind::Random, 1);
                let count2 = iteration_count(ctx, GeneratorKind::Random, 2);
                let iter = random_floats(count0).flat_map(move |f1: $fty| {
                    random_floats(count1).flat_map(move |f2: $fty| {
                        random_floats(count2).map(move |f3: $fty| (f1, f2, f3))
                    })
                });
                KnownSize::new(iter, count0 * count1 * count2)
            }
        }

        impl RandomInput for (i32, $fty) {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let count0 = iteration_count(ctx, GeneratorKind::Random, 0);
                let count1 = iteration_count(ctx, GeneratorKind::Random, 1);
                let range0 = int_range(ctx, GeneratorKind::Random, 0);
                let iter = random_ints(count0, range0)
                    .flat_map(move |f1: i32| random_floats(count1).map(move |f2: $fty| (f1, f2)));
                KnownSize::new(iter, count0 * count1)
            }
        }

        impl RandomInput for ($fty, i32) {
            fn get_cases(ctx: &CheckCtx) -> impl ExactSizeIterator<Item = Self> {
                let count0 = iteration_count(ctx, GeneratorKind::Random, 0);
                let count1 = iteration_count(ctx, GeneratorKind::Random, 1);
                let range1 = int_range(ctx, GeneratorKind::Random, 1);
                let iter = random_floats(count0).flat_map(move |f1: $fty| {
                    random_ints(count1, range1.clone()).map(move |f2: i32| (f1, f2))
                });
                KnownSize::new(iter, count0 * count1)
            }
        }
    };
}

#[cfg(f16_enabled)]
impl_random_input!(f16);
impl_random_input!(f32);
impl_random_input!(f64);
#[cfg(f128_enabled)]
impl_random_input!(f128);

/// Create a test case iterator.
pub fn get_test_cases<RustArgs: RandomInput>(
    ctx: &CheckCtx,
) -> impl Iterator<Item = RustArgs> + use<'_, RustArgs> {
    RustArgs::get_cases(ctx)
}
