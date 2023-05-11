#![allow(unused)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use rand::RngCore;

/// Copied from `std::test_helpers::test_rng`, since these tests rely on the
/// seed not being the same for every RNG invocation too.
#[track_caller]
pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
    use core::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    core::panic::Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    rand::SeedableRng::from_seed(seed)
}

// Copied from std::sys_common::io
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn join(&self, path: &str) -> PathBuf {
        let TempDir(ref p) = *self;
        p.join(path)
    }

    pub fn path(&self) -> &Path {
        let TempDir(ref p) = *self;
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Gee, seeing how we're testing the fs module I sure hope that we
        // at least implement this correctly!
        let TempDir(ref p) = *self;
        let result = fs::remove_dir_all(p);
        // Avoid panicking while panicking as this causes the process to
        // immediately abort, without displaying test results.
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

#[track_caller] // for `test_rng`
pub fn tmpdir() -> TempDir {
    let p = env::temp_dir();
    let mut r = test_rng();
    let ret = p.join(&format!("rust-{}", r.next_u32()));
    fs::create_dir(&ret).unwrap();
    TempDir(ret)
}
