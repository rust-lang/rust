// ignore-windows: Unwind panicking does not currently work on Windows
#![allow(arithmetic_overflow)]

fn main() {
    let _n = 1i64 >> 64;
}
