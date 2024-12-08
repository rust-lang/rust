//@ run-rustfix
// Issue #109429
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::hash::Hash;

pub struct Hash128_1;

impl BuildHasher for Hash128_1 {
    type Hasher = DefaultHasher;
    fn build_hasher(&self) -> DefaultHasher { DefaultHasher::default() }
}

#[allow(unused)]
pub fn hashmap_copy<T, U>(
    map: &HashMap<T, U, Hash128_1>,
) where T: Hash + Clone, U: Clone
{
    let mut copy: Vec<U> = map.clone().into_values().collect(); //~ ERROR
}

pub fn make_map() -> HashMap<String, i64, Hash128_1>
{
    HashMap::with_hasher(Hash128_1)
}
fn main() {}
