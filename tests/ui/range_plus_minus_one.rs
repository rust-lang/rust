// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f() -> usize {
    42
}

#[warn(clippy::range_plus_one)]
fn main() {
    for _ in 0..2 {}
    for _ in 0..=2 {}

    for _ in 0..3 + 1 {}
    for _ in 0..=3 + 1 {}

    for _ in 0..1 + 5 {}
    for _ in 0..=1 + 5 {}

    for _ in 1..1 + 1 {}
    for _ in 1..=1 + 1 {}

    for _ in 0..13 + 13 {}
    for _ in 0..=13 - 7 {}

    for _ in 0..(1 + f()) {}
    for _ in 0..=(1 + f()) {}

    let _ = ..11 - 1;
    let _ = ..=11 - 1;
    let _ = ..=(11 - 1);
    let _ = (1..11 + 1);
    let _ = (f() + 1)..(f() + 1);

    let mut vec: Vec<()> = std::vec::Vec::new();
    vec.drain(..);
}
