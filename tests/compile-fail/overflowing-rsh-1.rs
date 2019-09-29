#![allow(exceeding_bitshifts, const_err)]

fn main() {
    let _n = 1i64 >> 64; //~ ERROR attempt to shift right with overflow
}
