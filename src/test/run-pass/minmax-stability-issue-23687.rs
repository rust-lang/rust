// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core)]
use std::fmt::Debug;
use std::cmp::{self, PartialOrd, Ordering};
use std::iter::MinMaxResult::MinMax;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Foo {
    n: u8,
    name: &'static str
}

impl PartialOrd for Foo {
    fn partial_cmp(&self, other: &Foo) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Foo {
    fn cmp(&self, other: &Foo) -> Ordering {
        self.n.cmp(&other.n)
    }
}

fn main() {
    let a = Foo { n: 4, name: "a" };
    let b = Foo { n: 4, name: "b" };
    let c = Foo { n: 8, name: "c" };
    let d = Foo { n: 8, name: "d" };
    let e = Foo { n: 22, name: "e" };
    let f = Foo { n: 22, name: "f" };

    let data = [a, b, c, d, e, f];

    // `min` should return the left when the values are equal
    assert_eq!(data.iter().min(), Some(&a));
    assert_eq!(data.iter().min_by(|a| a.n), Some(&a));
    assert_eq!(cmp::min(a, b), a);
    assert_eq!(cmp::min(b, a), b);
    assert_eq!(cmp::partial_min(a, b), Some(a));
    assert_eq!(cmp::partial_min(b, a), Some(b));

    // `max` should return the right when the values are equal
    assert_eq!(data.iter().max(), Some(&f));
    assert_eq!(data.iter().max_by(|a| a.n), Some(&f));
    assert_eq!(cmp::max(e, f), f);
    assert_eq!(cmp::max(f, e), e);
    assert_eq!(cmp::partial_max(e, f), Some(f));
    assert_eq!(cmp::partial_max(f, e), Some(e));

    // Similar for `min_max`
    assert_eq!(data.iter().min_max(), MinMax(&a, &f));
    assert_eq!(data[1..5].iter().min_max(), MinMax(&b, &e));
    assert_eq!(data[2..4].iter().min_max(), MinMax(&c, &d));

    let mut presorted = data.to_vec();
    presorted.sort();
    assert_stable(&presorted);

    let mut presorted = data.to_vec();
    presorted.sort_by(|a, b| a.cmp(b));
    assert_stable(&presorted);

    // Assert that sorted and min/max are the same
    fn assert_stable<T: Ord + Debug>(presorted: &[T]) {
        for slice in presorted.windows(2) {
            let a = &slice[0];
            let b = &slice[1];

            assert_eq!(a, cmp::min(a, b));
            assert_eq!(b, cmp::max(a, b));
        }
    }
}
