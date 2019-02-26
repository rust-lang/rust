#![allow(exceeding_bitshifts)]

fn main() {
    let _n = 1i64 >> 64; //~ ERROR attempt to shift right with overflow
}
