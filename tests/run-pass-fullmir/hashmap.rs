use std::collections::{self, HashMap};
use std::hash::BuildHasherDefault;

fn main() {
    let map : HashMap<String, i32, BuildHasherDefault<collections::hash_map::DefaultHasher>> = Default::default();
    assert_eq!(map.values().fold(0, |x, y| x+y), 0);

    // TODO: This performs bit operations on the least significant bit of a pointer
//     for i in 0..33 {
//         map.insert(format!("key_{}", i), i);
//         assert_eq!(map.values().fold(0, |x, y| x+y), i*(i+1)/2);
//     }

    // TODO: Test Entry API
}
