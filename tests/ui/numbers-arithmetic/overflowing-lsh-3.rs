//@ build-fail
//@ compile-flags: -C debug-assertions

#![deny(arithmetic_overflow)]

fn main() {
    let _x = 1_u64 << 64;
    //~^ ERROR: this arithmetic operation will overflow
}
