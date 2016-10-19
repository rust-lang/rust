// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that .zip() specialization preserves side effects
// in sideeffectful iterator adaptors.

use std::cell::Cell;

#[derive(Debug)]
struct CountClone(Cell<i32>);

fn count_clone() -> CountClone { CountClone(Cell::new(0)) }

impl PartialEq<i32> for CountClone {
    fn eq(&self, rhs: &i32) -> bool {
        self.0.get() == *rhs
    }
}

impl Clone for CountClone {
    fn clone(&self) -> Self {
        let ret = CountClone(self.0.clone());
        let n = self.0.get();
        self.0.set(n + 1);
        ret
    }
}

fn test_zip_cloned_sideffectful() {
    let xs = [count_clone(), count_clone(), count_clone(), count_clone()];
    let ys = [count_clone(), count_clone()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) { }

    assert_eq!(&xs, &[1, 1, 1, 0][..]);
    assert_eq!(&ys, &[1, 1][..]);

    let xs = [count_clone(), count_clone()];
    let ys = [count_clone(), count_clone(), count_clone(), count_clone()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) { }

    assert_eq!(&xs, &[1, 1][..]);
    assert_eq!(&ys, &[1, 1, 0, 0][..]);
}

fn test_zip_map_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) { }

    assert_eq!(&xs, &[1, 1, 1, 1, 1, 0]);
    assert_eq!(&ys, &[1, 1, 1, 1]);

    let mut xs = [0; 4];
    let mut ys = [0; 6];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) { }

    assert_eq!(&xs, &[1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1, 0, 0]);
}

fn test_zip_map_rev_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        it.next_back();
    }
    assert_eq!(&xs, &[0, 0, 0, 1, 1, 1]);
    assert_eq!(&ys, &[0, 0, 0, 1]);

    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        (&mut it).take(5).count();
        it.next_back();
    }
    assert_eq!(&xs, &[1, 1, 1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1]);
}

fn test_zip_nested_sideffectful() {
    let mut xs = [0; 6];
    let ys = [0; 4];

    {
        // test that it has the side effect nested inside enumerate
        let it = xs.iter_mut().map(|x| *x = 1).enumerate().zip(&ys);
        it.count();
    }
    assert_eq!(&xs, &[1, 1, 1, 1, 1, 0]);
}

fn main() {
    test_zip_cloned_sideffectful();
    test_zip_map_sideffectful();
    test_zip_map_rev_sideffectful();
    test_zip_nested_sideffectful();
}
