// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr;

pub trait MoveMap<T>: Sized {
    fn move_map<F>(self, mut f: F) -> Self where F: FnMut(T) -> T {
        self.move_flat_map(|e| Some(f(e)))
    }

    fn move_flat_map<F, I>(self, f: F) -> Self
        where F: FnMut(T) -> I,
              I: IntoIterator<Item=T>;
}

impl<T> MoveMap<T> for Vec<T> {
    fn move_flat_map<F, I>(mut self, mut f: F) -> Self
        where F: FnMut(T) -> I,
              I: IntoIterator<Item=T>
    {
        let mut read_i = 0;
        let mut write_i = 0;
        unsafe {
            let mut old_len = self.len();
            self.set_len(0); // make sure we just leak elements in case of panic

            while read_i < old_len {
                // move the read_i'th item out of the vector and map it
                // to an iterator
                let e = ptr::read(self.get_unchecked(read_i));
                let mut iter = f(e).into_iter();
                read_i += 1;

                while let Some(e) = iter.next() {
                    if write_i < read_i {
                        ptr::write(self.get_unchecked_mut(write_i), e);
                        write_i += 1;
                    } else {
                        // If this is reached we ran out of space
                        // in the middle of the vector.
                        // However, the vector is in a valid state here,
                        // so we just do a somewhat inefficient insert.
                        self.set_len(old_len);
                        self.insert(write_i, e);

                        old_len = self.len();
                        self.set_len(0);

                        read_i += 1;
                        write_i += 1;
                    }
                }
            }

            // write_i tracks the number of actually written new items.
            self.set_len(write_i);
        }

        self
    }
}

impl<T> MoveMap<T> for ::ptr::P<[T]> {
    fn move_flat_map<F, I>(self, f: F) -> Self
        where F: FnMut(T) -> I,
              I: IntoIterator<Item=T>
    {
        ::ptr::P::from_vec(self.into_vec().move_flat_map(f))
    }
}
