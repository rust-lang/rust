//@ known-bug: #148614
use std::arch::global_asm;

static X: isize = 3;

global_asm! {
    "{}",
    sym X::<{}>,
}

fn main() {}
