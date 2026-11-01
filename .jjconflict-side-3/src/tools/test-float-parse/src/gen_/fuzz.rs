use std::any::{TypeId, type_name};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Mutex;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

use crate::{Float, Generator, Int, SEED};

/// Mapping of float types to the number of iterations that should be run.
///
/// We could probably make `Generator::new` take an argument instead of the global state,
/// but we only load this once so it works.
static FUZZ_COUNTS: Mutex<BTreeMap<TypeId, u64>> = Mutex::new(BTreeMap::new());

/// Generic fuzzer; just tests deterministic random bit patterns N times.
pub struct Fuzz<F> {
    iter: Range<u64>,
    rng: ChaCha8Rng,
    /// Allow us to use generics in `Iterator`.
    marker: PhantomData<F>,
}

impl<F: Float> Fuzz<F> {
    /// Register how many iterations the fuzzer should run for a type. Uses some logic by
    /// default, but if `from_cfg` is `Some`, that will be used instead.
    pub fn set_iterations(from_cfg: Option<u64>) {
        let count = if let Some(cfg_count) = from_cfg {
            cfg_count
        } else if F::BITS <= crate::MAX_BITS_FOR_EXHAUUSTIVE {
            // If we run exhaustively, still fuzz but only do half as many bits. The only goal here is
            // to catch failures from e.g. high bit patterns before exhaustive tests would get to them.
            (F::Int::MAX >> (F::BITS / 2)).try_into().unwrap()
        } else {
            // Eveything bigger gets a fuzz test with as many iterations as `f32` exhaustive.
            u32::MAX.into()
        };

        let _ = FUZZ_COUNTS.lock().unwrap().insert(TypeId::of::<F>(), count);
    }
}

impl<F: Float> Generator<F> for Fuzz<F>
where
    StandardUniform: Distribution<<F as Float>::Int>,
{
    const SHORT_NAME: &'static str = "fuzz";

    type WriteCtx = F;

    fn total_tests() -> u64 {
        *FUZZ_COUNTS
            .lock()
            .unwrap()
            .get(&TypeId::of::<F>())
            .unwrap_or_else(|| panic!("missing fuzz count for {}", type_name::<F>()))
    }

    fn new() -> Self {
        let rng = ChaCha8Rng::from_seed(SEED);

        Self { iter: 0..Self::total_tests(), rng, marker: PhantomData }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        write!(s, "{ctx:e}").unwrap();
    }
}

impl<F: Float> Iterator for Fuzz<F>
where
    StandardUniform: Distribution<<F as Float>::Int>,
{
    type Item = <Self as Generator<F>>::WriteCtx;

    fn next(&mut self) -> Option<Self::Item> {
        let _ = self.iter.next()?;
        let i: F::Int = self.rng.random();

        Some(F::from_bits(i))
    }
}
