// ignore-test
// compile-flags: --force-warns arithmetic_overflow
// check-pass

#![allow(arithmetic_overflow)]

fn main() {
    1_i32 << 32;
    //~^ WARN this arithmetic operation will overflow
}
