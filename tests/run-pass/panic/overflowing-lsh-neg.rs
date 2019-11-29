// ignore-windows: Unwind panicking does not currently work on Windows
#![allow(exceeding_bitshifts)]
#![allow(const_err)]

fn main() {
    let _n = 2i64 << -1;
}
