// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(libc)]

extern crate libc;

#[link(name = "test", kind = "static")]
extern {
    fn slice_len(s: &[u8]) -> libc::size_t;
    fn slice_elem(s: &[u8], idx: libc::size_t) -> u8;
}

fn main() {
    let data = [1,2,3,4,5];

    unsafe {
        assert_eq!(data.len(), slice_len(&data) as usize);
        assert_eq!(data[0], slice_elem(&data, 0));
        assert_eq!(data[1], slice_elem(&data, 1));
        assert_eq!(data[2], slice_elem(&data, 2));
        assert_eq!(data[3], slice_elem(&data, 3));
        assert_eq!(data[4], slice_elem(&data, 4));
    }
}
