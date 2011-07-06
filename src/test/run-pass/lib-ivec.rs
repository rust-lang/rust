// xfail-stage0

use std;
import std::ivec;
import std::option;
import std::option::none;
import std::option::some;

fn square(uint n) -> uint { ret n * n; }

fn square_alias(&uint n) -> uint { ret n * n; }

pred is_three(&uint n) -> bool { ret n == 3u; }

fn square_if_odd(&uint n) -> option::t[uint] {
    ret if (n % 2u == 1u) { some(n * n) } else { none };
}

fn add(&uint x, &uint y) -> uint { ret x + y; }

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
    // Test on-stack init_fn.
    auto v = ivec::init_fn(square, 3u);
    assert (ivec::len(v) == 3u);
    assert (v.(0) == 0u);
    assert (v.(1) == 1u);
    assert (v.(2) == 4u);

    // Test on-heap init_fn.
    v = ivec::init_fn(square, 5u);
    assert (ivec::len(v) == 5u);
    assert (v.(0) == 0u);
    assert (v.(1) == 1u);
    assert (v.(2) == 4u);
    assert (v.(3) == 9u);
    assert (v.(4) == 16u);
}

fn test_init_elt() {
    // Test on-stack init_elt.
    auto v = ivec::init_elt(10u, 2u);
    assert (ivec::len(v) == 2u);
    assert (v.(0) == 10u);
    assert (v.(1) == 10u);

    // Test on-heap init_elt.
    v = ivec::init_elt(20u, 6u);
    assert (v.(0) == 20u);
    assert (v.(1) == 20u);
    assert (v.(2) == 20u);
    assert (v.(3) == 20u);
    assert (v.(4) == 20u);
    assert (v.(5) == 20u);
}

fn test_last() {
    auto n = ivec::last(~[]);
    assert (n == none);
    n = ivec::last(~[ 1, 2, 3 ]);
    assert (n == some(3));
    n = ivec::last(~[ 1, 2, 3, 4, 5 ]);
    assert (n == some(5));
}

fn test_slice() {
    // Test on-stack -> on-stack slice.
    auto v = ivec::slice(~[ 1, 2, 3 ], 1u, 3u);
    assert (ivec::len(v) == 2u);
    assert (v.(0) == 2);
    assert (v.(1) == 3);

    // Test on-heap -> on-stack slice.
    v = ivec::slice(~[ 1, 2, 3, 4, 5 ], 0u, 3u);
    assert (ivec::len(v) == 3u);
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (v.(2) == 3);

    // Test on-heap -> on-heap slice.
    v = ivec::slice(~[ 1, 2, 3, 4, 5, 6 ], 1u, 6u);
    assert (ivec::len(v) == 5u);
    assert (v.(0) == 2);
    assert (v.(1) == 3);
    assert (v.(2) == 4);
    assert (v.(3) == 5);
    assert (v.(4) == 6);
}

fn test_pop() {
    // Test on-stack pop.
    auto v = ~[ 1, 2, 3 ];
    auto e = ivec::pop(v);
    assert (ivec::len(v) == 2u);
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (e == 3);

    // Test on-heap pop.
    v = ~[ 1, 2, 3, 4, 5 ];
    e = ivec::pop(v);
    assert (ivec::len(v) == 4u);
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (v.(2) == 3);
    assert (v.(3) == 4);
    assert (e == 5);
}

fn test_grow() {
    // Test on-stack grow().
    auto v = ~[];
    ivec::grow(v, 2u, 1);
    assert (ivec::len(v) == 2u);
    assert (v.(0) == 1);
    assert (v.(1) == 1);

    // Test on-heap grow().
    ivec::grow(v, 3u, 2);
    assert (ivec::len(v) == 5u);
    assert (v.(0) == 1);
    assert (v.(1) == 1);
    assert (v.(2) == 2);
    assert (v.(3) == 2);
    assert (v.(4) == 2);
}

fn test_grow_fn() {
    auto v = ~[];
    ivec::grow_fn(v, 3u, square);
    assert (ivec::len(v) == 3u);
    assert (v.(0) == 0u);
    assert (v.(1) == 1u);
    assert (v.(2) == 4u);
}

fn test_grow_set() {
    auto v = ~[ mutable 1, 2, 3 ];
    ivec::grow_set(v, 4u, 4, 5);
    assert (ivec::len(v) == 5u);
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (v.(2) == 3);
    assert (v.(3) == 4);
    assert (v.(4) == 5);
}

fn test_map() {
    // Test on-stack map.
    auto v = ~[ 1u, 2u, 3u ];
    auto w = ivec::map(square_alias, v);
    assert (ivec::len(w) == 3u);
    assert (w.(0) == 1u);
    assert (w.(1) == 4u);
    assert (w.(2) == 9u);

    // Test on-heap map.
    v = ~[ 1u, 2u, 3u, 4u, 5u ];
    w = ivec::map(square_alias, v);
    assert (ivec::len(w) == 5u);
    assert (w.(0) == 1u);
    assert (w.(1) == 4u);
    assert (w.(2) == 9u);
    assert (w.(3) == 16u);
    assert (w.(4) == 25u);
}

fn test_filter_map() {
    // Test on-stack filter-map.
    auto v = ~[ 1u, 2u, 3u ];
    auto w = ivec::filter_map(square_if_odd, v);
    assert (ivec::len(w) == 2u);
    assert (w.(0) == 1u);
    assert (w.(1) == 9u);

    // Test on-heap filter-map.
    v = ~[ 1u, 2u, 3u, 4u, 5u ];
    w = ivec::filter_map(square_if_odd, v);
    assert (ivec::len(w) == 3u);
    assert (w.(0) == 1u);
    assert (w.(1) == 9u);
    assert (w.(2) == 25u);
}

fn test_foldl() {
    // Test on-stack fold.
    auto v = ~[ 1u, 2u, 3u ];
    auto sum = ivec::foldl(add, 0u, v);
    assert (sum == 6u);

    // Test on-heap fold.
    v = ~[ 1u, 2u, 3u, 4u, 5u ];
    sum = ivec::foldl(add, 0u, v);
    assert (sum == 15u);
}

fn test_any_and_all() {
    assert (ivec::any(is_three, ~[ 1u, 2u, 3u ]));
    assert (!ivec::any(is_three, ~[ 0u, 1u, 2u ]));
    assert (ivec::any(is_three, ~[ 1u, 2u, 3u, 4u, 5u ]));
    assert (!ivec::any(is_three, ~[ 1u, 2u, 4u, 5u, 6u ]));

    assert (ivec::all(is_three, ~[ 3u, 3u, 3u ]));
    assert (!ivec::all(is_three, ~[ 3u, 3u, 2u ]));
    assert (ivec::all(is_three, ~[ 3u, 3u, 3u, 3u, 3u ]));
    assert (!ivec::all(is_three, ~[ 3u, 3u, 0u, 1u, 2u ]));
}

fn main() {
    test_reserve_and_on_heap();
    test_unsafe_ptrs();

    // Accessors
    test_init_fn();
    test_init_elt();
    test_last();
    test_slice();

    // Mutators
    test_pop();

    // Appending
    test_grow();
    test_grow_fn();
    test_grow_set();

    // Functional utilities
    test_map();
    test_filter_map();
    test_foldl();
    test_any_and_all();
}

