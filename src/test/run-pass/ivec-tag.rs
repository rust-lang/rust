// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

fn producer(c: &Chan<~[u8]>) {
    c.send(
         ~[1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8,
          13u8]);
}

pub fn main() {
    let (p, ch) = Chan::<~[u8]>::new();
    let _prod = task::spawn(proc() {
        producer(&ch)
    });

    let _data: ~[u8] = p.recv();
}
