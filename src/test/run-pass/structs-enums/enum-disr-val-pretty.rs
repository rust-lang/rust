// run-pass
#![allow(non_camel_case_types)]
// pp-exact


enum color { red = 1, green, blue, imaginary = -1, }

pub fn main() {
    test_color(color::red, 1, "red".to_string());
    test_color(color::green, 2, "green".to_string());
    test_color(color::blue, 3, "blue".to_string());
    test_color(color::imaginary, -1, "imaginary".to_string());
}

fn test_color(color: color, val: isize, _name: String) {
    assert_eq!(color as isize , val);
}
