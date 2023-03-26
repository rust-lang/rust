//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::collections::HashMap;
use std::hash::BuildHasher;

// Gather all references from a mutable iterator and make sure Miri notices if
// using them is dangerous.
fn test_all_refs<'a, T: 'a>(dummy: &mut T, iter: impl Iterator<Item = &'a mut T>) {
    // Gather all those references.
    let mut refs: Vec<&mut T> = iter.collect();
    // Use them all. Twice, to be sure we got all interleavings.
    for r in refs.iter_mut() {
        std::mem::swap(dummy, r);
    }
    for r in refs {
        std::mem::swap(dummy, r);
    }
}

fn smoketest_map<S: BuildHasher>(mut map: HashMap<i32, i32, S>) {
    map.insert(0, 0);
    assert_eq!(map.values().fold(0, |x, y| x + y), 0);

    let num = 25;
    for i in 1..num {
        map.insert(i, i);
    }
    assert_eq!(map.values().fold(0, |x, y| x + y), num * (num - 1) / 2); // check the right things are in the table now

    // Inserting again replaces the existing entries
    for i in 0..num {
        map.insert(i, num - 1 - i);
    }
    assert_eq!(map.values().fold(0, |x, y| x + y), num * (num - 1) / 2);

    test_all_refs(&mut 13, map.values_mut());
}

fn main() {
    // hashbrown uses Miri on its own CI; we just do a smoketest here.
    smoketest_map(HashMap::new());
}
