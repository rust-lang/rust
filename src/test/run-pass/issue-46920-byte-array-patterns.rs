// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const CURSOR_PARTITION_LABEL: &'static [u8] = b"partition";
const CURSOR_EVENT_TYPE_LABEL: &'static [u8] = b"event_type";
const BYTE_PATTERN: &'static [u8; 5] = b"hello";

fn match_slice(x: &[u8]) -> u32 {
    match x {
        CURSOR_PARTITION_LABEL => 0,
        CURSOR_EVENT_TYPE_LABEL => 1,
        _ => 2,
    }
}

fn match_array(x: &[u8; 5]) -> bool {
    match x {
        BYTE_PATTERN => true,
        _ => false
    }
}

fn main() {
    assert_eq!(match_slice(b"abcde"), 2);
    assert_eq!(match_slice(b"event_type"), 1);
    assert_eq!(match_slice(b"partition"), 0);

    assert_eq!(match_array(b"hello"), true);
    assert_eq!(match_array(b"hella"), false);
}
