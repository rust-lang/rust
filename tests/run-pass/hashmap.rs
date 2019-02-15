use std::collections::{self, HashMap};
use std::hash::BuildHasherDefault;

fn main() {
    let mut map : HashMap<i32, i32, BuildHasherDefault<collections::hash_map::DefaultHasher>> = Default::default();
    map.insert(0, 0);
    assert_eq!(map.values().fold(0, |x, y| x+y), 0);

    let table_base = map.get(&0).unwrap() as *const _;

    let num = 22; // large enough to trigger a resize
    for i in 1..num {
        map.insert(i, i);
    }
    assert!(table_base != map.get(&0).unwrap() as *const _); // make sure relocation happened
    assert_eq!(map.values().fold(0, |x, y| x+y), num*(num-1)/2); // check the right things are in the table now

    // Inserting again replaces the existing entries
    for i in 0..num {
        map.insert(i, num-1-i);
    }
    assert_eq!(map.values().fold(0, |x, y| x+y), num*(num-1)/2);

    // TODO: Test Entry API, Iterators, ...
}
