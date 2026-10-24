//@ run-pass
//@ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

use std::hint::black_box;

pub fn count(curr: u64, top: u64) -> u64 {
   if black_box(curr) >= top {
        curr
   } else {
        become count(curr + 1, top)
   }
}

fn main() {
    println!("{}", count(0, black_box(1000000)));
}
