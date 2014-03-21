// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that type assignability is used to search for instances when
// making method calls, but only if there aren't any matches without
// it.


trait iterable<A> {
    fn iterate(&self, blk: |x: &A| -> bool) -> bool;
}

impl<'a,A> iterable<A> for &'a [A] {
    fn iterate(&self, f: |x: &A| -> bool) -> bool {
        self.iter().advance(f)
    }
}

impl<A> iterable<A> for Vec<A> {
    fn iterate(&self, f: |x: &A| -> bool) -> bool {
        self.iter().advance(f)
    }
}

fn length<A, T: iterable<A>>(x: T) -> uint {
    let mut len = 0;
    x.iterate(|_y| {
        len += 1;
        true
    });
    return len;
}

pub fn main() {
    let x: Vec<int> = vec!(0,1,2,3);
    // Call a method
    x.iterate(|y| { assert!(*x.get(*y as uint) == *y); true });
    // Call a parameterized function
    assert_eq!(length(x.clone()), x.len());
    // Call a parameterized function, with type arguments that require
    // a borrow
    assert_eq!(length::<int, &[int]>(x.as_slice()), x.len());

    // Now try it with a type that *needs* to be borrowed
    let z = [0,1,2,3];
    // Call a method
    z.iterate(|y| { assert!(z[*y] == *y); true });
    // Call a parameterized function
    assert_eq!(length::<int, &[int]>(z), z.len());
}
