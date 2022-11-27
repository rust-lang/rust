// compile-flags: --test

#![allow(unused_mut, clippy::get_first, clippy::from_iter_instead_of_collect)]
#![warn(clippy::unwrap_used)]
#![deny(clippy::get_unwrap)]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;

struct GetFalsePositive {
    arr: [u32; 3],
}

impl GetFalsePositive {
    fn get(&self, pos: usize) -> Option<&u32> {
        self.arr.get(pos)
    }
    fn get_mut(&mut self, pos: usize) -> Option<&mut u32> {
        self.arr.get_mut(pos)
    }
}

fn main() {
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_slice = &mut [0, 1, 2, 3];
    let mut some_vec = vec![0, 1, 2, 3];
    let mut some_vecdeque: VecDeque<_> = some_vec.iter().cloned().collect();
    let mut some_hashmap: HashMap<u8, char> = HashMap::from_iter(vec![(1, 'a'), (2, 'b')]);
    let mut some_btreemap: BTreeMap<u8, char> = BTreeMap::from_iter(vec![(1, 'a'), (2, 'b')]);
    let mut false_positive = GetFalsePositive { arr: [0, 1, 2] };

    {
        // Test `get().unwrap()`
        let _ = boxed_slice.get(1).unwrap();
        let _ = some_slice.get(0).unwrap();
        let _ = some_vec.get(0).unwrap();
        let _ = some_vecdeque.get(0).unwrap();
        let _ = some_hashmap.get(&1).unwrap();
        let _ = some_btreemap.get(&1).unwrap();
        #[allow(clippy::unwrap_used)]
        let _ = false_positive.get(0).unwrap();
        // Test with deref
        let _: u8 = *boxed_slice.get(1).unwrap();
    }

    {
        // Test `get_mut().unwrap()`
        *boxed_slice.get_mut(0).unwrap() = 1;
        *some_slice.get_mut(0).unwrap() = 1;
        *some_vec.get_mut(0).unwrap() = 1;
        *some_vecdeque.get_mut(0).unwrap() = 1;
        // Check false positives
        #[allow(clippy::unwrap_used)]
        {
            *some_hashmap.get_mut(&1).unwrap() = 'b';
            *some_btreemap.get_mut(&1).unwrap() = 'b';
            *false_positive.get_mut(0).unwrap() = 1;
        }
    }

    {
        // Test `get().unwrap().foo()` and `get_mut().unwrap().bar()`
        let _ = some_vec.get(0..1).unwrap().to_vec();
        let _ = some_vec.get_mut(0..1).unwrap().to_vec();
    }
}

#[cfg(test)]
mod issue9612 {
    // should not lint in `#[cfg(test)]` modules
    #[test]
    fn test_fn() {
        let _a: u8 = 2.try_into().unwrap();
        let _a: u8 = 3.try_into().expect("");

        util();
    }

    fn util() {
        let _a: u8 = 4.try_into().unwrap();
        let _a: u8 = 5.try_into().expect("");
        // should still warn
        let _ = Box::new([0]).get(1).unwrap();
    }
}
