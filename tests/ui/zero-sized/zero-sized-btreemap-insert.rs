// run-pass
#![allow(unused_variables)]
#![allow(unused_imports)]
use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::BTreeMap;
use std::iter::Iterator;

#[derive(Eq, Hash, Debug, Ord, PartialEq, PartialOrd)]
struct Zst;

fn main() {
    const N: usize = 8;

    for len in 0..N {
        let mut tester = BTreeMap::new();
        assert_eq!(tester.len(), 0);
        for bit in 0..len {
            tester.insert(Zst, ());
        }
        assert_eq!(tester.len(), if len == 0 { 0 } else { 1 });
        assert_eq!(tester.iter().count(), if len == 0 { 0 } else { 1 });
        assert_eq!(tester.get(&Zst).is_some(), len > 0);
        tester.clear();
    }
}
