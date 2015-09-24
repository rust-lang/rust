// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(index_assign_trait)]
#![feature(indexed_assignments)]

use std::collections::HashMap;
use std::mem;
use std::ops::{Deref, DerefMut, Index, IndexAssign, Range, RangeFull};

fn main() {
    // test insertion via IndexAssign (no overloading)
    let mut map = Map::new();

    map[0] = 1;
    assert_eq!(map[0], 1);

    map[1] = 2;
    assert_eq!(map[1], 2);

    map[0] = 3;
    assert_eq!(map[0], 3);

    // test overloading the index
    let mut array = [0, 1, 2, 3];

    {
        let slice = Slice::new(&mut array);
        slice[..] = 0;
    }

    assert_eq!(array, [0, 0, 0, 0]);

    {
        let slice = Slice::new(&mut array);
        let rhs: &[_] = &[1, 2];
        slice[1..3] = rhs;
    }

    assert_eq!(array, [0, 1, 2, 0]);

    // test overloading the RHS
    {
        let slice = Slice::new(&mut array);
        slice[1..3] = 0;
    }

    assert_eq!(array, [0, 0, 0, 0]);

    // test through deref
    let mut v = Vector(vec![0, 0, 0, 0]);

    v[..] = 1;

    assert_eq!(v.0, [1, 1, 1, 1]);

    let rhs: &[_] = &[2, 3];
    v[1..3] = rhs;

    assert_eq!(v.0, [1, 2, 3, 1]);

    // test through proxy
    Slice::new(&mut array)[..] = 1;

    assert_eq!(array, [1, 1, 1, 1]);
}

struct Map(HashMap<u32, i32>);

impl Map {
    fn new() -> Map {
        Map(HashMap::new())
    }
}

impl Index<u32> for Map {
    type Output = i32;

    fn index(&self, idx: u32) -> &i32 {
        &self.0[&idx]
    }
}

impl IndexAssign<u32, i32> for Map {
    fn index_assign(&mut self, idx: u32, rhs: i32) {
        self.0.insert(idx, rhs);
    }
}

struct Vector(Vec<i32>);

impl Deref for Vector {
    type Target = Slice;

    fn deref(&self) -> &Slice {
        unsafe {
            mem::transmute(self.0.deref())
        }
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Slice {
        unsafe {
            mem::transmute(self.0.deref_mut())
        }
    }
}

struct Slice([i32]);

impl Slice {
    fn new(xs: &mut [i32]) -> &mut Slice {
        unsafe {
            mem::transmute(xs)
        }
    }
}

impl IndexAssign<RangeFull, i32> for Slice {
    fn index_assign(&mut self, _: RangeFull, rhs: i32) {
        for lhs in &mut self.0 {
            *lhs = rhs.clone()
        }
    }
}

impl<'a> IndexAssign<Range<usize>, &'a [i32]> for Slice {
    fn index_assign(&mut self, r: Range<usize>, rhs: &[i32]) {
        let lhs = &mut self.0[r];

        assert_eq!(lhs.len(), rhs.len());

        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = rhs.clone();
        }
    }
}

impl<'a> IndexAssign<Range<usize>, i32> for Slice {
    fn index_assign(&mut self, r: Range<usize>, rhs: i32) {
        for lhs in &mut self.0[r] {
            *lhs = rhs.clone();
        }
    }
}
