// xfail-stage0

use std;
import std::ivec;

fn test_reserve_and_on_heap() {
    let int[] v = ~[ 1, 2 ];
    assert (!ivec::on_heap(v));
    ivec::reserve(v, 8u);
    assert (ivec::on_heap(v));
}

fn test_unsafe_ptrs() {
    // Test on-stack copy-from-buf.
    auto a = ~[ 1, 2, 3 ];
    auto ptr = ivec::to_ptr(a);
    auto b = ~[];
    ivec::unsafe::copy_from_buf(b, ptr, 3u);
    assert (ivec::len(b) == 3u);
    assert (b.(0) == 1);
    assert (b.(1) == 2);
    assert (b.(2) == 3);

    // Test on-heap copy-from-buf.
    auto c = ~[ 1, 2, 3, 4, 5 ];
    ptr = ivec::to_ptr(c);
    auto d = ~[];
    ivec::unsafe::copy_from_buf(d, ptr, 5u);
    assert (ivec::len(d) == 5u);
    assert (d.(0) == 1);
    assert (d.(1) == 2);
    assert (d.(2) == 3);
    assert (d.(3) == 4);
    assert (d.(4) == 5);
}

fn test_init_fn() {
    fn square(uint n) -> uint { ret n * n; }
    auto v = ivec::init_fn(square, 3u);
    assert (ivec::len(v) == 3u);
    assert (v.(0) == 1u);
    assert (v.(1) == 4u);
    assert (v.(2) == 9u);
}

fn main() {
    test_reserve_and_on_heap();
    //test_unsafe_ptrs();
    //test_init_fn();
}

