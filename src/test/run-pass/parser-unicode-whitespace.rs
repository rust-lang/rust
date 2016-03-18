// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Beware editing: it has numerous whitespace characters which are important.
// It contains one ranges from the 'PATTERN_WHITE_SPACE' property outlined in
// http://unicode.org/Public/UNIDATA/PropList.txt
//
// The characters in the first expression of the assertion can be generated
// from: "4\u{0C}+\n\t\r7\t*\u{20}2\u{85}/\u{200E}3\u{200F}*\u{2028}2\u{2029}"
pub fn main() {
assert_eq!(4+

7   * 2/‎3‏* 2 , 4 + 7 * 2 / 3 * 2);
}
