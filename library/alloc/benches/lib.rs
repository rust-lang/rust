// Disabling on android for the time being
// See https://github.com/rust-lang/rust/issues/73535#event-3477699747
#![cfg(not(target_os = "android"))]
#![feature(btree_extract_if)]
#![feature(iter_next_chunk)]
#![feature(repr_simd)]
#![feature(slice_partition_dedup)]
#![feature(strict_provenance)]
#![feature(test)]
#![deny(fuzzy_provenance_casts)]

extern crate test;

mod binary_heap;
mod btree;
mod linked_list;
mod slice;
mod str;
mod string;
mod vec;
mod vec_deque;

/// Returns a `rand::Rng` seeded with a consistent seed.
///
/// This is done to avoid introducing nondeterminism in benchmark results.
fn bench_rng() -> rand_xorshift::XorShiftRng {
    const SEED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    rand::SeedableRng::from_seed(SEED)
}
