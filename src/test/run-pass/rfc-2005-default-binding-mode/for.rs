// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

pub fn main() {
    let mut tups = vec![(0u8, 1u8)];

    for (n, m) in &tups {
        let _: &u8 = n;
        let _: &u8 = m;
    }

    for (n, m) in &mut tups {
        *n += 1;
        *m += 2;
    }

    assert_eq!(tups, vec![(1u8, 3u8)]);

    for (n, m) in tups {
        println!("{} {}", m, n);
    }
}
