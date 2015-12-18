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

pub trait MoveMap {
    type Item;
    fn move_map<F>(self, f: F) -> Self
        where F: FnMut(Self::Item) -> Self::Item;
}

pub trait MoveFlatMap {
    type Item;
    fn move_flat_map<F, I>(self, f: F) -> Self
        where F: FnMut(Self::Item) -> I,
              I: IntoIterator<Item = Self::Item>;
}

impl<Container, T> MoveMap for Container
    where for<'a> &'a mut Container: IntoIterator<Item = &'a mut T>
{
    type Item = T;
    fn move_map<F>(mut self, mut f: F) -> Container where F: FnMut(T) -> T {
        for p in &mut self {
            unsafe {
                // FIXME(#5016) this shouldn't need to zero to be safe.
                ptr::write(p, f(ptr::read_and_drop(p)));
            }
        }
        self
    }
}

impl<T> MoveFlatMap for Vec<T> {
    type Item = T;
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

impl<T> MoveFlatMap for ::ptr::P<[T]> {
    type Item = T;
    fn move_flat_map<F, I>(self, f: F) -> Self
        where F: FnMut(T) -> I,
              I: IntoIterator<Item=T>
    {
        let v: Vec<_> = self.into();
        v.move_flat_map(f).into()
    }
}
