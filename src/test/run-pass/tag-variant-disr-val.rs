// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum color {
    red = 0xff0000,
    green = 0x00ff00,
    blue = 0x0000ff,
    black = 0x000000,
    white = 0xFFFFFF,
    imaginary = -1,
    purple = 1 << 1,
    orange = 8 >> 1
}

impl Eq for color {
    fn eq(&self, other: &color) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    fn ne(&self, other: &color) -> bool { !(*self).eq(other) }
}

pub fn main() {
    test_color(red, 0xff0000, ~"red");
    test_color(green, 0x00ff00, ~"green");
    test_color(blue, 0x0000ff, ~"blue");
    test_color(black, 0x000000, ~"black");
    test_color(white, 0xFFFFFF, ~"white");
    test_color(imaginary, -1, ~"imaginary");
    test_color(purple, 2, ~"purple");
    test_color(orange, 4, ~"orange");
}

fn test_color(color: color, val: int, name: ~str) {
    //assert!(unsafe::transmute(color) == val);
    assert_eq!(color as int, val);
    assert_eq!(color as float, val as float);
    assert!(get_color_alt(color) == name);
    assert!(get_color_if(color) == name);
}

fn get_color_alt(color: color) -> ~str {
    match color {
      red => {~"red"}
      green => {~"green"}
      blue => {~"blue"}
      black => {~"black"}
      white => {~"white"}
      imaginary => {~"imaginary"}
      purple => {~"purple"}
      orange => {~"orange"}
    }
}

fn get_color_if(color: color) -> ~str {
    if color == red {~"red"}
    else if color == green {~"green"}
    else if color == blue {~"blue"}
    else if color == black {~"black"}
    else if color == white {~"white"}
    else if color == imaginary {~"imaginary"}
    else if color == purple {~"purple"}
    else if color == orange {~"orange"}
    else {~"unknown"}
}
