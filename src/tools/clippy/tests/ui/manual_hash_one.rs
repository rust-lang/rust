#![warn(clippy::manual_hash_one)]
#![allow(clippy::needless_borrows_for_generic_args)]

use std::hash::{BuildHasher, Hash, Hasher};

fn returned(b: impl BuildHasher) -> u64 {
    let mut hasher = b.build_hasher();
    true.hash(&mut hasher);
    hasher.finish()
    //~^ manual_hash_one
}

fn unsized_receiver(b: impl BuildHasher, s: &str) {
    let mut hasher = b.build_hasher();
    s[4..10].hash(&mut hasher);
    let _ = hasher.finish();
    //~^ manual_hash_one
}

fn owned_value(b: impl BuildHasher, v: Vec<u32>) -> Vec<u32> {
    let mut hasher = b.build_hasher();
    v.hash(&mut hasher);
    let _ = hasher.finish();
    //~^ manual_hash_one
    v
}

fn reused_hasher(b: impl BuildHasher) {
    let mut hasher = b.build_hasher();
    true.hash(&mut hasher);
    let _ = hasher.finish();
    let _ = hasher.finish();
}

fn reused_hasher_in_return(b: impl BuildHasher) -> u64 {
    let mut hasher = b.build_hasher();
    true.hash(&mut hasher);
    let _ = hasher.finish();
    hasher.finish()
}

fn no_hash(b: impl BuildHasher) {
    let mut hasher = b.build_hasher();
    let _ = hasher.finish();
}

fn hash_twice(b: impl BuildHasher) {
    let mut hasher = b.build_hasher();
    true.hash(&mut hasher);
    true.hash(&mut hasher);
    let _ = hasher.finish();
}

fn other_hasher(b: impl BuildHasher) {
    let mut other_hasher = b.build_hasher();

    let mut hasher = b.build_hasher();
    true.hash(&mut other_hasher);
    let _ = hasher.finish();
}

fn finish_then_hash(b: impl BuildHasher) {
    let mut hasher = b.build_hasher();
    let _ = hasher.finish();
    true.hash(&mut hasher);
}

fn in_macro(b: impl BuildHasher) {
    macro_rules! m {
        ($b:expr) => {{
            let mut hasher = $b.build_hasher();
            true.hash(&mut hasher);
            let _ = hasher.finish();
        }};
    }

    m!(b);
}

#[clippy::msrv = "1.70"]
fn msrv_1_70(b: impl BuildHasher, v: impl Hash) {
    let mut hasher = b.build_hasher();
    v.hash(&mut hasher);
    let _ = hasher.finish();
}

#[clippy::msrv = "1.71"]
fn msrv_1_71(b: impl BuildHasher, v: impl Hash) {
    let mut hasher = b.build_hasher();
    v.hash(&mut hasher);
    let _ = hasher.finish();
    //~^ manual_hash_one
}
