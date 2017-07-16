// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Deref, DerefMut};

struct CheckedDeref<T, F> {
    value: T,
    check: F
}

impl<F: Fn(&T) -> bool, T> Deref for CheckedDeref<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        assert!((self.check)(&self.value));
        &self.value
    }
}

impl<F: Fn(&T) -> bool, T> DerefMut for CheckedDeref<T, F> {
    fn deref_mut(&mut self) -> &mut T {
        assert!((self.check)(&self.value));
        &mut self.value
    }
}


fn main() {
    let mut v = CheckedDeref {
        value: vec![0],
        check: |v: &Vec<_>| !v.is_empty()
    };
    v.push(1);
    assert_eq!(*v, vec![0, 1]);
}
