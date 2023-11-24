//@compile-flags: -Zmiri-strict-provenance
use rand::prelude::*;

// Test using the `rand` crate to generate randomness.
fn main() {
    // Fully deterministic seeding.
    let mut rng = SmallRng::seed_from_u64(42);
    let _val = rng.gen::<i32>();
    let _val = rng.gen::<isize>();
    let _val = rng.gen::<i128>();

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
