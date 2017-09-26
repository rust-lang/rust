// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod builders;
mod float;
mod num;

#[test]
fn test_format_flags() {
    // No residual flags left by pointer formatting
    let p = "".as_ptr();
    assert_eq!(format!("{:p} {:x}", p, 16), format!("{:p} 10", p));

    assert_eq!(format!("{: >3}", 'a'), "  a");
}

#[test]
fn test_pointer_formats_data_pointer() {
    let b: &[u8] = b"";
    let s: &str = "";
    assert_eq!(format!("{:p}", s), format!("{:p}", s.as_ptr()));
    assert_eq!(format!("{:p}", b), format!("{:p}", b.as_ptr()));
}

#[test]
fn test_estimated_capacity() {
    assert_eq!(format_args!("").estimated_capacity(), 0);
    assert_eq!(format_args!("{}", "").estimated_capacity(), 0);
    assert_eq!(format_args!("Hello").estimated_capacity(), 5);
    assert_eq!(format_args!("Hello, {}!", "").estimated_capacity(), 16);
    assert_eq!(format_args!("{}, hello!", "World").estimated_capacity(), 0);
    assert_eq!(format_args!("{}. 16-bytes piece", "World").estimated_capacity(), 32);
}

#[test]
fn test_additional_slice_formatting() {
    let a: &[u8] = &[1, 2, 7, 8, 120, 255];
    let b: &[f32] = &[3e5, 5e7];

    // Test all hex forms
    assert_eq!(format!("{:x}", a),    "[1, 2, 7, 8, 78, ff]");
    assert_eq!(format!("{:X}", a),    "[1, 2, 7, 8, 78, FF]");
    assert_eq!(format!("{:02x}", a),  "[01, 02, 07, 08, 78, ff]");
    assert_eq!(format!("{:#x}", a),   "[0x1, 0x2, 0x7, 0x8, 0x78, 0xff]");
    assert_eq!(format!("{:#04x}", a), "[0x01, 0x02, 0x07, 0x08, 0x78, 0xff]");

    // Superficial tests for the rest of the forms
    assert_eq!(format!("{:o}", a), "[1, 2, 7, 10, 170, 377]");
    assert_eq!(format!("{:b}", a), "[1, 10, 111, 1000, 1111000, 11111111]");
    assert_eq!(format!("{:e}", b), "[3e5, 5e7]");
    assert_eq!(format!("{:E}", b), "[3E5, 5E7]");
}
