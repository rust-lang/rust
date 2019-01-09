// compile-pass
// compile-flags: -Z continue-parse-after-error

#![feature(box_syntax)]

use std::fmt::Debug;

fn main() {
    let x: Box<Debug+> = box 3 as Box<Debug+>; // Trailing `+` is OK
}
