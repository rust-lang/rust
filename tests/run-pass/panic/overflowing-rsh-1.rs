// ignore-windows: Unwind panicking does not currently work on Windows
#![allow(exceeding_bitshifts, const_err)]

fn main() {
    let _n = 1i64 >> 64;
}
