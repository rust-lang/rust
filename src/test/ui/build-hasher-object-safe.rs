// run-pass

use std::hash::BuildHasher;
use std::collections::hash_map::{DefaultHasher, RandomState};

fn ensure_object_safe(_: &dyn BuildHasher<Hasher = DefaultHasher>) {}

fn main() {
    ensure_object_safe(&RandomState::new());
}
