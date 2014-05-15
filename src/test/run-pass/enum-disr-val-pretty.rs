// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact

enum color { red = 1, green, blue, imaginary = -1, }

pub fn main() {
    test_color(red, 1, "red".to_strbuf());
    test_color(green, 2, "green".to_strbuf());
    test_color(blue, 3, "blue".to_strbuf());
    test_color(imaginary, -1, "imaginary".to_strbuf());
}

fn test_color(color: color, val: int, _name: StrBuf) {
    assert!(color as int == val);
    assert!(color as f64 == val as f64);
}
