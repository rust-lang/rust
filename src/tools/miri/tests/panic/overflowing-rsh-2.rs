#![allow(arithmetic_overflow)]

fn main() {
    // Make sure we catch overflows that would be hidden by first casting the RHS to u32
    let _n = 1i64 >> (u32::MAX as i64 + 1);
}
