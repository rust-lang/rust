// ignore-windows: Unwind panicking does not currently work on Windows
#![allow(unconditional_panic)]

fn main() {
    let _n = 1 / 0;
}
