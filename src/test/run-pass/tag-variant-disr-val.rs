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
    test_color(red, 0xff0000, "red".to_owned());
    test_color(green, 0x00ff00, "green".to_owned());
    test_color(blue, 0x0000ff, "blue".to_owned());
    test_color(black, 0x000000, "black".to_owned());
    test_color(white, 0xFFFFFF, "white".to_owned());
    test_color(imaginary, -1, "imaginary".to_owned());
    test_color(purple, 2, "purple".to_owned());
    test_color(orange, 4, "orange".to_owned());
}

fn test_color(color: color, val: int, name: ~str) {
    //assert!(unsafe::transmute(color) == val);
    assert_eq!(color as int, val);
    assert_eq!(color as f64, val as f64);
    assert!(get_color_alt(color) == name);
    assert!(get_color_if(color) == name);
}

fn get_color_alt(color: color) -> ~str {
    match color {
      red => {"red".to_owned()}
      green => {"green".to_owned()}
      blue => {"blue".to_owned()}
      black => {"black".to_owned()}
      white => {"white".to_owned()}
      imaginary => {"imaginary".to_owned()}
      purple => {"purple".to_owned()}
      orange => {"orange".to_owned()}
    }
}

fn get_color_if(color: color) -> ~str {
    if color == red {"red".to_owned()}
    else if color == green {"green".to_owned()}
    else if color == blue {"blue".to_owned()}
    else if color == black {"black".to_owned()}
    else if color == white {"white".to_owned()}
    else if color == imaginary {"imaginary".to_owned()}
    else if color == purple {"purple".to_owned()}
    else if color == orange {"orange".to_owned()}
    else {"unknown".to_owned()}
}
