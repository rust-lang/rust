// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that variables and loop labels with the same name don't
// shadow each other.

// Tests that loop labels honor their lexical scopes and don't clash with
// each other.

#[allow(unreachable_code)];

pub fn main() {
    let mut outer = 0i;
    'bar: loop {
        let mut bar = 0i;
        'bar: for _ in range(0, 1) {
            bar += 1;
            // This should refer to the inner loop
            continue 'bar;
            unreachable!();
        }
        assert_eq!(bar, 1);
        outer += 1;
        // This should break out of the outer loop
        break 'bar;

        'bar: loop {
            unreachable!();
        }
    }
    assert_eq!(outer, 1);
}
