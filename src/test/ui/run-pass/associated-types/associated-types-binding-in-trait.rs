// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a case where the associated type binding (to `bool`, in this
// case) is derived from the trait definition. Issue #21636.


use std::vec;

pub trait BitIter {
    type Iter: Iterator<Item=bool>;
    fn bit_iter(self) -> <Self as BitIter>::Iter;
}

impl BitIter for Vec<bool> {
    type Iter = vec::IntoIter<bool>;
    fn bit_iter(self) -> <Self as BitIter>::Iter {
        self.into_iter()
    }
}

fn count<T>(arg: T) -> usize
    where T: BitIter
{
    let mut sum = 0;
    for i in arg.bit_iter() {
        if i {
            sum += 1;
        }
    }
    sum
}

fn main() {
    let v = vec![true, false, true];
    let c = count(v);
    assert_eq!(c, 2);
}
