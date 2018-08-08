// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// compile-pass

#![feature(nll)]

fn fibs(n: u32) -> impl Iterator<Item=u128> {
    (0 .. n)
    .scan((0, 1), |st, _| {
        *st = (st.1, st.0 + st.1);
        Some(*st)
    })
    .map(&|(f, _)| f)
}

fn main() {
    println!("{:?}", fibs(10).collect::<Vec<_>>());
}
