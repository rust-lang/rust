// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::LinkedList;
use std::iter::Iterator;

fn main() {
    const N: usize = 8;

    // Test that for all possible sequences of push_front / push_back,
    // we end up with a LinkedList of the correct size

    for len in 0..N {
        let mut tester = LinkedList::new();
        assert_eq!(tester.len(), 0);
        assert_eq!(tester.front(), None);
        for case in 0..(1 << len) {
            assert_eq!(tester.len(), 0);
            for bit in 0..len {
                if case & (1 << bit) != 0 {
                    tester.push_front(());
                } else {
                    tester.push_back(());
                }
            }
            assert_eq!(tester.len(), len);
            assert_eq!(tester.iter().count(), len);
            tester.clear();
        }
    }
}
