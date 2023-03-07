// run-pass
#![allow(non_camel_case_types)]

use color::{red, green, blue, black, white, imaginary, purple, orange};

#[derive(Copy, Clone)]
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

impl PartialEq for color {
    fn eq(&self, other: &color) -> bool {
        ((*self) as usize) == ((*other) as usize)
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

fn test_color(color: color, val: isize, name: String) {
    //assert_eq!(unsafe::transmute(color), val);
    assert_eq!(color as isize, val);
    assert_eq!(get_color_alt(color), name);
    assert_eq!(get_color_if(color), name);
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
