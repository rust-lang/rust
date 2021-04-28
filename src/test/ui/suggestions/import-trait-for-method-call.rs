use std::hash::BuildHasher;

fn next_u64() -> u64 {
    let bh = std::collections::hash_map::RandomState::new();
    let h = bh.build_hasher();
    h.finish() //~ ERROR no method named `finish` found for struct `DefaultHasher`
}

fn main() {}
