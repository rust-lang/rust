use std;
use std::bitv::*;

fn bitv_test() -> bool {
    let v1 = ~Bitv(31, false);
    let v2 = ~Bitv(31, true);
    v1.union(v2);
    true
}

fn main() {
    do iter::repeat(10000) || {bitv_test()};
}
