// build-pass

#![allow(incomplete_features)]
#![feature(inline_const)]

use std::cell::Cell;

fn main() {
    let _x = [const { Cell::new(0i32) }; 20];
}
