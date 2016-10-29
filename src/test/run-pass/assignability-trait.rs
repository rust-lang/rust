// Copyright 2012-4 The Rust Project Developers. See the COPYRIGHT
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
    fn iterate<F>(&self, blk: F) -> bool where F: FnMut(&A) -> bool;
}

impl<'a,A> iterable<A> for &'a [A] {
    fn iterate<F>(&self, f: F) -> bool where F: FnMut(&A) -> bool {
        self.iter().all(f)
    }
}

impl<A> iterable<A> for Vec<A> {
    fn iterate<F>(&self, f: F) -> bool where F: FnMut(&A) -> bool {
        self.iter().all(f)
    }
}

fn length<A, T: iterable<A>>(x: T) -> usize {
    let mut len = 0;
    x.iterate(|_y| {
        len += 1;
        true
    });
    return len;
}

pub fn main() {
    let x: Vec<isize> = vec![0,1,2,3];
    // Call a method
    x.iterate(|y| { assert_eq!(x[*y as usize], *y); true });
    // Call a parameterized function
    assert_eq!(length(x.clone()), x.len());
    // Call a parameterized function, with type arguments that require
    // a borrow
    assert_eq!(length::<isize, &[isize]>(&*x), x.len());

    // Now try it with a type that *needs* to be borrowed
    let z = [0,1,2,3];
    // Call a parameterized function
    assert_eq!(length::<isize, &[isize]>(&z), z.len());
}
