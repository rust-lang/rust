// ignore-windows: Unwind panicking does not currently work on Windows
#![allow(arithmetic_overflow)]

fn main() {
    let _n = 2i64 << -1;
}
