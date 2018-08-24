// pretty-expanded FIXME #23616

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(dead_code, unused_variables)]
#![feature(box_syntax)]

// Tests that the new `box` syntax works with unique pointers.

use std::boxed::Box;

struct Structure {
    x: isize,
    y: isize,
}

pub fn main() {
    let y: Box<isize> = box 2;
    let b: Box<isize> = box (1 + 2);
    let c = box (3 + 4);

    let s: Box<Structure> = box Structure {
        x: 3,
        y: 4,
    };
}
