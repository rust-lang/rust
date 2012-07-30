use std;
import std::bitv::*;

fn bitv_test() -> bool {
    let v1 = ~bitv(31, false);
    let v2 = ~bitv(31, true);
    v1.union(v2);
    true
}

fn main() {
    do iter::repeat(10000) || {bitv_test()};
}