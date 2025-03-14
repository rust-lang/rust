use std::env;
use std::str::FromStr;
use std::sync::OnceLock;

use rand::distr::Uniform;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

use crate::sort::zipf::ZipfDistribution;

/// Provides a set of patterns useful for testing and benchmarking sorting algorithms.
/// Currently limited to i32 values.

// --- Public ---

pub fn random(len: usize) -> Vec<i32> {
    //     .
    // : . : :
    // :.:::.::

    random_vec(len)
}

pub fn random_uniform<R>(len: usize, range: R) -> Vec<i32>
where
    Uniform<i32>: TryFrom<R, Error: std::fmt::Debug>,
{
    // :.:.:.::

    let mut rng: XorShiftRng = rand::SeedableRng::seed_from_u64(get_or_init_rand_seed());

    // Abstracting over ranges in Rust :(
    let dist = Uniform::try_from(range).unwrap();
    (0..len).map(|_| dist.sample(&mut rng)).collect()
}

pub fn random_zipf(len: usize, exponent: f64) -> Vec<i32> {
    // https://en.wikipedia.org/wiki/Zipf's_law

    let mut rng: XorShiftRng = rand::SeedableRng::seed_from_u64(get_or_init_rand_seed());

    // Abstracting over ranges in Rust :(
    let dist = ZipfDistribution::new(len, exponent).unwrap();
    (0..len).map(|_| dist.sample(&mut rng) as i32).collect()
}

pub fn random_sorted(len: usize, sorted_percent: f64) -> Vec<i32> {
    //     .:
    //   .:::. :
    // .::::::.::
    // [----][--]
    //  ^      ^
    //  |      |
    // sorted  |
    //     unsorted

    // Simulate pre-existing sorted slice, where len - sorted_percent are the new unsorted values
    // and part of the overall distribution.
    let mut v = random_vec(len);
    let sorted_len = ((len as f64) * (sorted_percent / 100.0)).round() as usize;

    v[0..sorted_len].sort_unstable();

    v
}

pub fn all_equal(len: usize) -> Vec<i32> {
    // ......
    // ::::::

    (0..len).map(|_| 66).collect::<Vec<_>>()
}

pub fn ascending(len: usize) -> Vec<i32> {
    //     .:
    //   .:::
    // .:::::

    (0..len as i32).collect::<Vec<_>>()
}

pub fn descending(len: usize) -> Vec<i32> {
    // :.
    // :::.
    // :::::.

    (0..len as i32).rev().collect::<Vec<_>>()
}

pub fn saw_mixed(len: usize, saw_count: usize) -> Vec<i32> {
    // :.  :.    .::.    .:
    // :::.:::..::::::..:::

    if len == 0 {
        return Vec::new();
    }

    let mut vals = random_vec(len);
    let chunks_size = len / saw_count.max(1);
    let saw_directions = random_uniform((len / chunks_size) + 1, 0..=1);

    for (i, chunk) in vals.chunks_mut(chunks_size).enumerate() {
        if saw_directions[i] == 0 {
            chunk.sort_unstable();
        } else if saw_directions[i] == 1 {
            chunk.sort_unstable_by_key(|&e| std::cmp::Reverse(e));
        } else {
            unreachable!();
        }
    }

    vals
}

pub fn saw_mixed_range(len: usize, range: std::ops::Range<usize>) -> Vec<i32> {
    //     :.
    // :.  :::.    .::.      .:
    // :::.:::::..::::::..:.:::

    // ascending and descending randomly picked, with length in `range`.

    if len == 0 {
        return Vec::new();
    }

    let mut vals = random_vec(len);

    let max_chunks = len / range.start;
    let saw_directions = random_uniform(max_chunks + 1, 0..=1);
    let chunk_sizes = random_uniform(max_chunks + 1, (range.start as i32)..(range.end as i32));

    let mut i = 0;
    let mut l = 0;
    while l < len {
        let chunk_size = chunk_sizes[i] as usize;
        let chunk_end = std::cmp::min(l + chunk_size, len);
        let chunk = &mut vals[l..chunk_end];

        if saw_directions[i] == 0 {
            chunk.sort_unstable();
        } else if saw_directions[i] == 1 {
            chunk.sort_unstable_by_key(|&e| std::cmp::Reverse(e));
        } else {
            unreachable!();
        }

        i += 1;
        l += chunk_size;
    }

    vals
}

pub fn pipe_organ(len: usize) -> Vec<i32> {
    //   .:.
    // .:::::.

    let mut vals = random_vec(len);

    let first_half = &mut vals[0..(len / 2)];
    first_half.sort_unstable();

    let second_half = &mut vals[(len / 2)..len];
    second_half.sort_unstable_by_key(|&e| std::cmp::Reverse(e));

    vals
}

pub fn get_or_init_rand_seed() -> u64 {
    *SEED_VALUE.get_or_init(|| {
        env::var("OVERRIDE_SEED")
            .ok()
            .map(|seed| u64::from_str(&seed).unwrap())
            .unwrap_or_else(rand_root_seed)
    })
}

// --- Private ---

static SEED_VALUE: OnceLock<u64> = OnceLock::new();

#[cfg(not(miri))]
fn rand_root_seed() -> u64 {
    // Other test code hashes `panic::Location::caller()` and constructs a seed from that, in these
    // tests we want to have a fuzzer like exploration of the test space, if we used the same caller
    // based construction we would always test the same.
    //
    // Instead we use the seconds since UNIX epoch / 10, given CI log output this value should be
    // reasonably easy to re-construct.

    use std::time::{SystemTime, UNIX_EPOCH};

    let epoch_seconds = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    epoch_seconds / 10
}

#[cfg(miri)]
fn rand_root_seed() -> u64 {
    // Miri is usually run with isolation with gives us repeatability but also permutations based on
    // other code that runs before.
    use core::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = std::hash::RandomState::new().build_hasher();
    core::panic::Location::caller().hash(&mut hasher);
    hasher.finish()
}

fn random_vec(len: usize) -> Vec<i32> {
    let mut rng: XorShiftRng = rand::SeedableRng::seed_from_u64(get_or_init_rand_seed());
    (0..len).map(|_| rng.random::<i32>()).collect()
}
