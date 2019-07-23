use std::collections::{self, HashMap};
use std::hash::{BuildHasherDefault, BuildHasher};

fn test_map<S: BuildHasher>(mut map: HashMap<i32, i32, S>) {
    map.insert(0, 0);
    assert_eq!(map.values().fold(0, |x, y| x+y), 0);

    let num = 25;
    for i in 1..num {
        map.insert(i, i);
    }
    assert_eq!(map.values().fold(0, |x, y| x+y), num*(num-1)/2); // check the right things are in the table now

    // Inserting again replaces the existing entries
    for i in 0..num {
        map.insert(i, num-1-i);
    }
    assert_eq!(map.values().fold(0, |x, y| x+y), num*(num-1)/2);

    // TODO: Test Entry API, Iterators, ...

}

fn main() {
    if cfg!(target_os = "macos") { // TODO: Implement libstd HashMap seeding for macOS (https://github.com/rust-lang/miri/issues/686).
        // Until then, use a deterministic map.
        test_map::<BuildHasherDefault<collections::hash_map::DefaultHasher>>(HashMap::default());
    } else {
        test_map(HashMap::new());
    }
}
