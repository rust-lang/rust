//@ needs-asm-support

use std::arch::global_asm;

fn main() {}

// Constants must be... constant
fn non_const_fn(x: i32) -> i32 { x }

global_asm!("/* {} */", const non_const_fn(0));
//~^ERROR: cannot call non-const function
