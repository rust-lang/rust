#![allow(exceeding_bitshifts, const_err)]

fn main() {
    // Make sure we catch overflows that would be hidden by first casting the RHS to u32
    let _n = 1i64 >> (u32::max_value() as i64 + 1); //~ ERROR attempt to shift right with overflow
}
