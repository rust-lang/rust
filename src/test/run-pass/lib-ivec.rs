// xfail-stage0

use std;
import std::ivec;

fn test_reserve_and_on_heap() {
    let int[] v = ~[ 1, 2 ];
    assert (!ivec::on_heap(v));
    ivec::reserve(v, 8u);
    assert (ivec::on_heap(v));
}

fn main() {
    test_reserve_and_on_heap();
}

