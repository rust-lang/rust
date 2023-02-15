//! We don't use `rand`, as that's too many things for us.
//!
//! We currently use oorandom instead, but it's missing these two utilities.
//! Perhaps we should switch to `fastrand`, or our own small PRNG, it's not like
//! we need anything more complicated than xor-shift.

pub fn shuffle<T>(slice: &mut [T], mut rand_index: impl FnMut(usize) -> usize) {
    let mut remaining = slice.len() - 1;
    while remaining > 0 {
        let index = rand_index(remaining);
        slice.swap(remaining, index);
        remaining -= 1;
    }
}

pub fn seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    RandomState::new().build_hasher().finish()
}
