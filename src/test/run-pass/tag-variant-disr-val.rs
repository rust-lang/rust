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
    test_color(red, 0xff0000, "red".to_string());
    test_color(green, 0x00ff00, "green".to_string());
    test_color(blue, 0x0000ff, "blue".to_string());
    test_color(black, 0x000000, "black".to_string());
    test_color(white, 0xFFFFFF, "white".to_string());
    test_color(imaginary, -1, "imaginary".to_string());
    test_color(purple, 2, "purple".to_string());
    test_color(orange, 4, "orange".to_string());
}

fn test_color(color: color, val: int, name: String) {
    //assert!(unsafe::transmute(color) == val);
    assert_eq!(color as int, val);
    assert_eq!(color as f64, val as f64);
    assert!(get_color_alt(color) == name);
    assert!(get_color_if(color) == name);
}

fn get_color_alt(color: color) -> String {
    match color {
      red => {"red".to_string()}
      green => {"green".to_string()}
      blue => {"blue".to_string()}
      black => {"black".to_string()}
      white => {"white".to_string()}
      imaginary => {"imaginary".to_string()}
      purple => {"purple".to_string()}
      orange => {"orange".to_string()}
    }
}

fn get_color_if(color: color) -> String {
    if color == red {"red".to_string()}
    else if color == green {"green".to_string()}
    else if color == blue {"blue".to_string()}
    else if color == black {"black".to_string()}
    else if color == white {"white".to_string()}
    else if color == imaginary {"imaginary".to_string()}
    else if color == purple {"purple".to_string()}
    else if color == orange {"orange".to_string()}
    else {"unknown".to_string()}
}
