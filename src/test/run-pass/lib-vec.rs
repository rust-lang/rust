
use std;

fn test_init_elt() {
    let vec[uint] v = std::vec::init_elt[uint](5u, 3u);
    assert (std::vec::len[uint](v) == 3u);
    assert (v.(0) == 5u);
    assert (v.(1) == 5u);
    assert (v.(2) == 5u);
}

fn id(uint x) -> uint { ret x; }

fn test_init_fn() {
    let fn(uint) -> uint  op = id;
    let vec[uint] v = std::vec::init_fn[uint](op, 5u);
    assert (std::vec::len[uint](v) == 5u);
    assert (v.(0) == 0u);
    assert (v.(1) == 1u);
    assert (v.(2) == 2u);
    assert (v.(3) == 3u);
    assert (v.(4) == 4u);
}

fn test_slice() {
    let vec[int] v = [1, 2, 3, 4, 5];
    auto v2 = std::vec::slice[int](v, 2u, 4u);
    assert (std::vec::len[int](v2) == 2u);
    assert (v2.(0) == 3);
    assert (v2.(1) == 4);
}

fn test_map() {
    fn square(&int x) -> int { ret x * x; }
    let std::option::operator[int, int] op = square;
    let vec[int] v = [1, 2, 3, 4, 5];
    let vec[int] s = std::vec::map[int, int](op, v);
    let int i = 0;
    while (i < 5) { assert (v.(i) * v.(i) == s.(i)); i += 1; }
}

fn test_map2() {
    fn times(&int x, &int y) -> int { ret x * y; }
    auto f = times;
    auto v0 = [1, 2, 3, 4, 5];
    auto v1 = [5, 4, 3, 2, 1];
    auto u = std::vec::map2[int, int, int](f, v0, v1);
    auto i = 0;
    while (i < 5) { assert (v0.(i) * v1.(i) == u.(i)); i += 1; }
}

fn test_filter_map() {
    fn halve(&int i) -> std::option::t[int] {
        if (i % 2 == 0) {
            ret std::option::some[int](i / 2);
        } else { ret std::option::none[int]; }
    }
    fn halve_for_sure(&int i) -> int { ret i / 2; }
    let vec[int] all_even = [0, 2, 8, 6];
    let vec[int] all_odd1 = [1, 7, 3];
    let vec[int] all_odd2 = [];
    let vec[int] mix = [9, 2, 6, 7, 1, 0, 0, 3];
    let vec[int] mix_dest = [1, 3, 0, 0];
    assert (std::vec::filter_map(halve, all_even) ==
                std::vec::map(halve_for_sure, all_even));
    assert (std::vec::filter_map(halve, all_odd1) == std::vec::empty[int]());
    assert (std::vec::filter_map(halve, all_odd2) == std::vec::empty[int]());
    assert (std::vec::filter_map(halve, mix) == mix_dest);
}

fn main() {
    test_init_elt();
    test_init_fn();
    test_slice();
    test_map();
    test_map2();
    test_filter_map();
}