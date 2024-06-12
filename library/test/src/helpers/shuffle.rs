use std::hash::{DefaultHasher, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::TestOpts;
use crate::types::{TestDescAndFn, TestId, TestName};

pub fn get_shuffle_seed(opts: &TestOpts) -> Option<u64> {
    opts.shuffle_seed.or_else(|| {
        if opts.shuffle {
            Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Failed to get system time")
                    .as_nanos() as u64,
            )
        } else {
            None
        }
    })
}

pub fn shuffle_tests(shuffle_seed: u64, tests: &mut [(TestId, TestDescAndFn)]) {
    let test_names: Vec<&TestName> = tests.iter().map(|test| &test.1.desc.name).collect();
    let test_names_hash = calculate_hash(&test_names);
    let mut rng = Rng::new(shuffle_seed, test_names_hash);
    shuffle(&mut rng, tests);
}

// `shuffle` is from `rust-analyzer/src/cli/analysis_stats.rs`.
fn shuffle<T>(rng: &mut Rng, slice: &mut [T]) {
    for i in 0..slice.len() {
        randomize_first(rng, &mut slice[i..]);
    }

    fn randomize_first<T>(rng: &mut Rng, slice: &mut [T]) {
        assert!(!slice.is_empty());
        let idx = rng.rand_range(0..slice.len() as u64) as usize;
        slice.swap(0, idx);
    }
}

struct Rng {
    state: u64,
    extra: u64,
}

impl Rng {
    fn new(seed: u64, extra: u64) -> Self {
        Self { state: seed, extra }
    }

    fn rand_range(&mut self, range: core::ops::Range<u64>) -> u64 {
        self.rand_u64() % (range.end - range.start) + range.start
    }

    fn rand_u64(&mut self) -> u64 {
        self.state = calculate_hash(&(self.state, self.extra));
        self.state
    }
}

// `calculate_hash` is from `core/src/hash/mod.rs`.
fn calculate_hash<T: core::hash::Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
