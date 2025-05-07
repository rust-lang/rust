//@ run-pass

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

#![allow(dead_code, unused_variables)]

// Tests that the new `box` syntax works with unique pointers.

struct Structure {
    x: isize,
    y: isize,
}

pub fn main() {
    let y: Box<isize> = Box::new(2);
    let b: Box<isize> = Box::new(1 + 2);
    let c = Box::new(3 + 4);

    let s: Box<Structure> = Box::new(Structure {
        x: 3,
        y: 4,
    });
}
