//@compile-flags: -Zmiri-strict-provenance
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn main() {
    // Test `getrandom` directly.
    let mut data = vec![0; 16];
    getrandom::getrandom(&mut data).unwrap();

    // Try seeding with "real" entropy.
    let mut rng = SmallRng::from_entropy();
    let _val = rng.gen::<i32>();
    let _val = rng.gen::<isize>();
    let _val = rng.gen::<i128>();

    // Also try per-thread RNG.
    let mut rng = rand::thread_rng();
    let _val = rng.gen::<i32>();
    let _val = rng.gen::<isize>();
    let _val = rng.gen::<i128>();
}
