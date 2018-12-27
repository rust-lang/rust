extern crate rand;

mod _common;

use std::char;
use rand::{IsaacRng, Rng, SeedableRng};
use rand::distributions::{Range, Sample};
use _common::{validate, SEED};

fn main() {
    let mut rnd = IsaacRng::from_seed(&SEED);
    let mut range = Range::new(0, 10);
    for _ in 0..5_000_000u64 {
        let num_digits = rnd.gen_range(100, 400);
        let digits = gen_digits(num_digits, &mut range, &mut rnd);
        validate(&digits);
    }
}

fn gen_digits<R: Rng>(n: u32, range: &mut Range<u32>, rnd: &mut R) -> String {
    let mut s = String::new();
    for _ in 0..n {
        let digit = char::from_digit(range.sample(rnd), 10).unwrap();
        s.push(digit);
    }
    s
}
