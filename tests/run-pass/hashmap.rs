// compile-flags: -Zmiri-seed=0000000000000000

use std::collections::{self, HashMap};
use std::hash::{BuildHasherDefault, BuildHasher};

fn test_map<S: BuildHasher>(mut map: HashMap<i32, i32, S>) {
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

fn main() {
    if cfg!(target_os = "macos") { // TODO: Implement random number generation on OS X.
        // Until then, use a deterministic map.
        let map : HashMap<i32, i32, BuildHasherDefault<collections::hash_map::DefaultHasher>> = HashMap::default();
        test_map(map);
    } else {
        let map: HashMap<i32, i32> = HashMap::default();
        test_map(map);
    }
}
