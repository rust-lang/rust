use rand::{RngCore, SeedableRng};

use crate::hash::{BuildHasher, Hash, Hasher, RandomState};
use crate::panic::Location;
use crate::path::{Path, PathBuf};
use crate::{env, fs, thread};

/// Test-only replacement for `rand::thread_rng()`, which is unusable for
/// us, as we want to allow running stdlib tests on tier-3 targets which may
/// not have `getrandom` support.
///
/// Does a bit of a song and dance to ensure that the seed is different on
/// each call (as some tests sadly rely on this), but doesn't try that hard.
///
/// This is duplicated in the `core`, `alloc` test suites (as well as
/// `std`'s integration tests), but figuring out a mechanism to share these
/// seems far more painful than copy-pasting a 7 line function a couple
/// times, given that even under a perma-unstable feature, I don't think we
/// want to expose types from `rand` from `std`.
#[track_caller]
pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
    let mut hasher = RandomState::new().build_hasher();
    Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    SeedableRng::from_seed(seed)
}

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
