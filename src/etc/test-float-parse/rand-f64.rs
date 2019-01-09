extern crate rand;

mod _common;

use _common::{validate, SEED};
use rand::{IsaacRng, Rng, SeedableRng};
use std::mem::transmute;

fn main() {
    let mut rnd = IsaacRng::from_seed(&SEED);
    let mut i = 0;
    while i < 10_000_000 {
        let bits = rnd.next_u64();
        let x: f64 = unsafe { transmute(bits) };
        if x.is_finite() {
            validate(&format!("{:e}", x));
            i += 1;
        }
    }
}
