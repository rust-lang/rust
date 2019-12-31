// macOS needs FS access for its HashMap:
// compile-flags: -Zmiri-disable-isolation

use std::collections::HashMap;
use std::hash::BuildHasher;

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
    test_map(HashMap::new());
}
