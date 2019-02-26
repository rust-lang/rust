#![allow(exceeding_bitshifts)]
#![allow(const_err)]

fn main() {
    let _n = 2i64 << -1; //~ ERROR attempt to shift left with overflow
}
