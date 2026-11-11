//@ revisions: bpass1 bpass2 bpass3
//@ compile-flags: -Coverflow-checks=on
//@ ignore-backends: gcc

#![warn(arithmetic_overflow)]

fn main() {
    let _ = 255u8 + 1; //~ WARNING operation will overflow
}
