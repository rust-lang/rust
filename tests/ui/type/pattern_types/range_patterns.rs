#![feature(pattern_types)]
#![allow(incomplete_features)]

// check-pass

fn main() {
    let x: u32 is 1.. = todo!();
}
