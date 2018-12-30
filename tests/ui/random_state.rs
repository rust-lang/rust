#![warn(clippy::random_state)]

use std::collections::hash_map::RandomState;
use std::collections::hash_map::{DefaultHasher, HashMap};
use std::hash::{BuildHasherDefault};

fn main() {
    // Should warn
    let mut map = HashMap::new();
    map.insert(3, 4);
    let mut map = HashMap::with_hasher(RandomState::new());
    map.insert(true, false);
    let _map: HashMap<_, _> = vec![(2, 3)].into_iter().collect();
    let _vec: Vec<HashMap<i32, i32>>;
    // Shouldn't warn
    let _map: HashMap<i32, i32, BuildHasherDefault<DefaultHasher>> = HashMap::default();
    let mut map = HashMap::with_hasher(BuildHasherDefault::<DefaultHasher>::default());
    map.insert("a", "b");
}
