// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::copy_iterator)]

#[derive(Copy, Clone)]
struct Countdown(u8);

impl Iterator for Countdown {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        self.0.checked_sub(1).map(|c| {
            self.0 = c;
            c
        })
    }
}

fn main() {
    let my_iterator = Countdown(5);
    let a: Vec<_> = my_iterator.take(1).collect();
    assert_eq!(a.len(), 1);
    let b: Vec<_> = my_iterator.collect();
    assert_eq!(b.len(), 5);
}
