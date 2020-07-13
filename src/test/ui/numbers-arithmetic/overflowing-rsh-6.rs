// build-fail
// compile-flags: -C debug-assertions

#![deny(arithmetic_overflow, const_err)]

fn main() {
    let _n = 1i64 >> [64][0];
    //~^ ERROR: this arithmetic operation will overflow
}
