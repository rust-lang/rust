#![deny(unsafe_code)]

core::arch::global_asm! { "nop" } //~ ERROR: 3:1: 3:34: usage of `global_asm` is unsafe [unsafe_code]

fn main() {

}
