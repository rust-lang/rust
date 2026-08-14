//@ known-bug: #148628
use std::arch::asm;

static TEST: i32 = 0;

pub unsafe fn bar<T>() {
    asm!("/* {0} */", sym TEST::<T>);
}

fn main() {}
