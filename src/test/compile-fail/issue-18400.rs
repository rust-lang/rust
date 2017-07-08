// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Set<T> {
    fn contains(&self, _: T) -> bool;
    fn set(&mut self, _: T);
}

impl<'a, T, S> Set<&'a [T]> for S where
    T: Copy,
    S: Set<T>,
{
    fn contains(&self, bits: &[T]) -> bool {
        bits.iter().all(|&bit| self.contains(bit))
    }

    fn set(&mut self, bits: &[T]) {
        for &bit in bits {
            self.set(bit)
        }
    }
}

fn main() {
    let bits: &[_] = &[0, 1];

    0.contains(bits);
    //~^ ERROR overflow
}
