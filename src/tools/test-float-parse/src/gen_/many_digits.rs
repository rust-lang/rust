use std::char;
use std::fmt::Write;
use std::marker::PhantomData;
use std::ops::{Range, RangeInclusive};

use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{Float, Generator, SEED};

/// Total iterations
const ITERATIONS: u64 = 5_000_000;

/// Possible lengths of the string, excluding decimals and exponents
const POSSIBLE_NUM_DIGITS: RangeInclusive<usize> = 100..=400;

/// Range of possible exponents
const EXP_RANGE: Range<i32> = -4500..4500;

/// Try strings of random digits.
pub struct RandDigits<F> {
    rng: ChaCha8Rng,
    iter: Range<u64>,
    uniform: Uniform<u32>,
    /// Allow us to use generics in `Iterator`.
    marker: PhantomData<F>,
}

impl<F: Float> Generator<F> for RandDigits<F> {
    const NAME: &'static str = "random digits";

    const SHORT_NAME: &'static str = "rand digits";

    type WriteCtx = String;

    fn total_tests() -> u64 {
        ITERATIONS
    }

    fn new() -> Self {
        let rng = ChaCha8Rng::from_seed(SEED);
        let range = Uniform::try_from(0..10).unwrap();

        Self { rng, iter: 0..ITERATIONS, uniform: range, marker: PhantomData }
    }

    fn write_string(s: &mut String, ctx: Self::WriteCtx) {
        *s = ctx;
    }
}

impl<F: Float> Iterator for RandDigits<F> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let _ = self.iter.next()?;
        let num_digits = self.rng.random_range(POSSIBLE_NUM_DIGITS);
        let has_decimal = self.rng.random_bool(0.2);
        let has_exp = self.rng.random_bool(0.2);

        let dec_pos = if has_decimal { Some(self.rng.random_range(0..num_digits)) } else { None };

        let mut s = String::with_capacity(num_digits);

        for pos in 0..num_digits {
            let digit = char::from_digit(self.uniform.sample(&mut self.rng), 10).unwrap();
            s.push(digit);

            if let Some(dec_pos) = dec_pos {
                if pos == dec_pos {
                    s.push('.');
                }
            }
        }

        if has_exp {
            let exp = self.rng.random_range(EXP_RANGE);
            write!(s, "e{exp}").unwrap();
        }

        Some(s)
    }
}
