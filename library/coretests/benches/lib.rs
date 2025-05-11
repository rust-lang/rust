// wasm32 does not support benches (no time).
#![cfg(not(target_arch = "wasm32"))]
// Disabling in Miri as these would take too long.
#![cfg(not(miri))]
#![feature(flt2dec)]
#![feature(test)]
#![feature(trusted_random_access)]
#![feature(iter_array_chunks)]
#![feature(iter_next_chunk)]
#![feature(iter_advance_by)]

extern crate test;

mod any;
mod array;
mod ascii;
mod char;
mod fmt;
mod hash;
mod iter;
mod net;
mod num;
mod ops;
mod pattern;
mod slice;
mod str;
mod tuple;

/// Returns a `rand::Rng` seeded with a consistent seed.
///
/// This is done to avoid introducing nondeterminism in benchmark results.
fn bench_rng() -> rand_xorshift::XorShiftRng {
    const SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    rand::SeedableRng::from_seed(SEED)
}
