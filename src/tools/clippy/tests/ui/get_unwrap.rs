//@run-rustfix

#![allow(
    unused_mut,
    clippy::from_iter_instead_of_collect,
    clippy::get_first,
    clippy::useless_vec
)]
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
mod issue9909 {
    #![allow(clippy::identity_op, clippy::unwrap_used, dead_code)]

    fn reduced() {
        let f = &[1, 2, 3];

        // include a borrow in the suggestion, even if the argument is not just a numeric literal
        let _x: &i32 = f.get(1 + 2).unwrap();

        // don't include a borrow here
        let _x = f.get(1 + 2).unwrap().to_string();

        // don't include a borrow here
        let _x = f.get(1 + 2).unwrap().abs();
    }

    // original code:
    fn linidx(row: usize, col: usize) -> usize {
        row * 1 + col * 3
    }

    fn main_() {
        let mut mat = [1.0f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];

        for i in 0..2 {
            for j in i + 1..3 {
                if mat[linidx(j, 3)] > mat[linidx(i, 3)] {
                    for k in 0..4 {
                        let (x, rest) = mat.split_at_mut(linidx(i, k) + 1);
                        let a = x.last_mut().unwrap();
                        let b = rest.get_mut(linidx(j, k) - linidx(i, k) - 1).unwrap();
                        ::std::mem::swap(a, b);
                    }
                }
            }
        }
        assert_eq!([9.0, 5.0, 1.0, 10.0, 6.0, 2.0, 11.0, 7.0, 3.0, 12.0, 8.0, 4.0], mat);
    }
}
