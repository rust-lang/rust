// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

fn slice_pat() {
    let sl: &[u8] = b"foo";

    match sl {
        [first, remainder..] => {
            let _: &u8 = first;
            assert_eq!(first, &b'f');
            assert_eq!(remainder, b"oo");
        }
        [] => panic!(),
    }
}

fn slice_pat_omission() {
     match &[0, 1, 2] {
        [..] => {}
     };
}

fn main() {
    slice_pat();
    slice_pat_omission();
}
