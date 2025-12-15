#![deny(black_box_zst_calls)]

use std::hint::black_box;

fn add(a: u32, b: u32) -> u32 {
    a + b
}

fn main() {
    let add_bb = black_box(add);
    //~^ ERROR `black_box` on zero-sized callable
    let _ = add_bb(1, 2);
}
