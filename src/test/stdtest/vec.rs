
use std;
import std::vec;
import std::vec::*;
import std::option;
import std::option::none;
import std::option::some;

fn square(n: uint) -> uint { ret n * n; }

fn square_alias(n: &uint) -> uint { ret n * n; }

pure fn is_three(n: &uint) -> bool { ret n == 3u; }

fn square_if_odd(n: &uint) -> option::t<uint> {
    ret if n % 2u == 1u { some(n * n) } else { none };
}

fn add(x: &uint, y: &uint) -> uint { ret x + y; }

#[test]
fn test_reserve_and_on_heap() {
    let v: [int] = [1, 2];
    assert (!vec::on_heap(v));
    vec::reserve(v, 8u);
    assert (vec::on_heap(v));
}

#[test]
fn test_unsafe_ptrs() {
    // Test on-stack copy-from-buf.
    let a = [1, 2, 3];
    let ptr = vec::to_ptr(a);
    let b = [];
    vec::unsafe::copy_from_buf(b, ptr, 3u);
    assert (vec::len(b) == 3u);
    assert (b[0] == 1);
    assert (b[1] == 2);
    assert (b[2] == 3);

    // Test on-heap copy-from-buf.
    let c = [1, 2, 3, 4, 5];
    ptr = vec::to_ptr(c);
    let d = [];
    vec::unsafe::copy_from_buf(d, ptr, 5u);
    assert (vec::len(d) == 5u);
    assert (d[0] == 1);
    assert (d[1] == 2);
    assert (d[2] == 3);
    assert (d[3] == 4);
    assert (d[4] == 5);
}

#[test]
fn test_init_fn() {
    // Test on-stack init_fn.
    let v = vec::init_fn(square, 3u);
    assert (vec::len(v) == 3u);
    assert (v[0] == 0u);
    assert (v[1] == 1u);
    assert (v[2] == 4u);

    // Test on-heap init_fn.
    v = vec::init_fn(square, 5u);
    assert (vec::len(v) == 5u);
    assert (v[0] == 0u);
    assert (v[1] == 1u);
    assert (v[2] == 4u);
    assert (v[3] == 9u);
    assert (v[4] == 16u);
}

#[test]
fn test_init_elt() {
    // Test on-stack init_elt.
    let v = vec::init_elt(10u, 2u);
    assert (vec::len(v) == 2u);
    assert (v[0] == 10u);
    assert (v[1] == 10u);

    // Test on-heap init_elt.
    v = vec::init_elt(20u, 6u);
    assert (v[0] == 20u);
    assert (v[1] == 20u);
    assert (v[2] == 20u);
    assert (v[3] == 20u);
    assert (v[4] == 20u);
    assert (v[5] == 20u);
}

#[test]
fn test_is_empty() {
    assert (vec::is_empty::<int>([]));
    assert (!vec::is_empty([0]));
}

#[test]
fn test_is_not_empty() {
    assert (vec::is_not_empty([0]));
    assert (!vec::is_not_empty::<int>([]));
}

#[test]
fn test_head() {
    let a = [11, 12];
    check (vec::is_not_empty(a));
    assert (vec::head(a) == 11);
}

#[test]
fn test_tail() {
    let a = [11];
    check (vec::is_not_empty(a));
    assert (vec::tail(a) == []);

    a = [11, 12];
    check (vec::is_not_empty(a));
    assert (vec::tail(a) == [12]);
}

#[test]
fn test_last() {
    let n = vec::last([]);
    assert (n == none);
    n = vec::last([1, 2, 3]);
    assert (n == some(3));
    n = vec::last([1, 2, 3, 4, 5]);
    assert (n == some(5));
}

#[test]
fn test_slice() {
    // Test on-stack -> on-stack slice.
    let v = vec::slice([1, 2, 3], 1u, 3u);
    assert (vec::len(v) == 2u);
    assert (v[0] == 2);
    assert (v[1] == 3);

    // Test on-heap -> on-stack slice.
    v = vec::slice([1, 2, 3, 4, 5], 0u, 3u);
    assert (vec::len(v) == 3u);
    assert (v[0] == 1);
    assert (v[1] == 2);
    assert (v[2] == 3);

    // Test on-heap -> on-heap slice.
    v = vec::slice([1, 2, 3, 4, 5, 6], 1u, 6u);
    assert (vec::len(v) == 5u);
    assert (v[0] == 2);
    assert (v[1] == 3);
    assert (v[2] == 4);
    assert (v[3] == 5);
    assert (v[4] == 6);
}

#[test]
fn test_pop() {
    // Test on-stack pop.
    let v = [1, 2, 3];
    let e = vec::pop(v);
    assert (vec::len(v) == 2u);
    assert (v[0] == 1);
    assert (v[1] == 2);
    assert (e == 3);

    // Test on-heap pop.
    v = [1, 2, 3, 4, 5];
    e = vec::pop(v);
    assert (vec::len(v) == 4u);
    assert (v[0] == 1);
    assert (v[1] == 2);
    assert (v[2] == 3);
    assert (v[3] == 4);
    assert (e == 5);
}

#[test]
fn test_grow() {
    // Test on-stack grow().
    let v = [];
    vec::grow(v, 2u, 1);
    assert (vec::len(v) == 2u);
    assert (v[0] == 1);
    assert (v[1] == 1);

    // Test on-heap grow().
    vec::grow(v, 3u, 2);
    assert (vec::len(v) == 5u);
    assert (v[0] == 1);
    assert (v[1] == 1);
    assert (v[2] == 2);
    assert (v[3] == 2);
    assert (v[4] == 2);
}

#[test]
fn test_grow_fn() {
    let v = [];
    vec::grow_fn(v, 3u, square);
    assert (vec::len(v) == 3u);
    assert (v[0] == 0u);
    assert (v[1] == 1u);
    assert (v[2] == 4u);
}

#[test]
fn test_grow_set() {
    let v = [mutable 1, 2, 3];
    vec::grow_set(v, 4u, 4, 5);
    assert (vec::len(v) == 5u);
    assert (v[0] == 1);
    assert (v[1] == 2);
    assert (v[2] == 3);
    assert (v[3] == 4);
    assert (v[4] == 5);
}

#[test]
fn test_map() {
    // Test on-stack map.
    let v = [1u, 2u, 3u];
    let w = vec::map(square_alias, v);
    assert (vec::len(w) == 3u);
    assert (w[0] == 1u);
    assert (w[1] == 4u);
    assert (w[2] == 9u);

    // Test on-heap map.
    v = [1u, 2u, 3u, 4u, 5u];
    w = vec::map(square_alias, v);
    assert (vec::len(w) == 5u);
    assert (w[0] == 1u);
    assert (w[1] == 4u);
    assert (w[2] == 9u);
    assert (w[3] == 16u);
    assert (w[4] == 25u);
}

#[test]
fn test_map2() {
    fn times(x: &int, y: &int) -> int { ret x * y; }
    let f = times;
    let v0 = [1, 2, 3, 4, 5];
    let v1 = [5, 4, 3, 2, 1];
    let u = vec::map2::<int, int, int>(f, v0, v1);
    let i = 0;
    while i < 5 { assert (v0[i] * v1[i] == u[i]); i += 1; }
}

#[test]
fn test_filter_map() {
    // Test on-stack filter-map.
    let v = [1u, 2u, 3u];
    let w = vec::filter_map(square_if_odd, v);
    assert (vec::len(w) == 2u);
    assert (w[0] == 1u);
    assert (w[1] == 9u);

    // Test on-heap filter-map.
    v = [1u, 2u, 3u, 4u, 5u];
    w = vec::filter_map(square_if_odd, v);
    assert (vec::len(w) == 3u);
    assert (w[0] == 1u);
    assert (w[1] == 9u);
    assert (w[2] == 25u);

    fn halve(i: &int) -> option::t<int> {
        if i % 2 == 0 {
            ret option::some::<int>(i / 2);
        } else { ret option::none::<int>; }
    }
    fn halve_for_sure(i: &int) -> int { ret i / 2; }
    let all_even: [int] = [0, 2, 8, 6];
    let all_odd1: [int] = [1, 7, 3];
    let all_odd2: [int] = [];
    let mix: [int] = [9, 2, 6, 7, 1, 0, 0, 3];
    let mix_dest: [int] = [1, 3, 0, 0];
    assert (filter_map(halve, all_even) == map(halve_for_sure, all_even));
    assert (filter_map(halve, all_odd1) == []);
    assert (filter_map(halve, all_odd2) == []);
    assert (filter_map(halve, mix) == mix_dest);

}

#[test]
fn test_foldl() {
    // Test on-stack fold.
    let v = [1u, 2u, 3u];
    let sum = vec::foldl(add, 0u, v);
    assert (sum == 6u);

    // Test on-heap fold.
    v = [1u, 2u, 3u, 4u, 5u];
    sum = vec::foldl(add, 0u, v);
    assert (sum == 15u);
}

#[test]
fn test_any_and_all() {
    assert (vec::any(is_three, [1u, 2u, 3u]));
    assert (!vec::any(is_three, [0u, 1u, 2u]));
    assert (vec::any(is_three, [1u, 2u, 3u, 4u, 5u]));
    assert (!vec::any(is_three, [1u, 2u, 4u, 5u, 6u]));

    assert (vec::all(is_three, [3u, 3u, 3u]));
    assert (!vec::all(is_three, [3u, 3u, 2u]));
    assert (vec::all(is_three, [3u, 3u, 3u, 3u, 3u]));
    assert (!vec::all(is_three, [3u, 3u, 0u, 1u, 2u]));
}

#[test]
fn test_zip_unzip() {
    let v1 = [1, 2, 3];
    let v2 = [4, 5, 6];
    let z1 = vec::zip(v1, v2);

    assert ((1, 4) == z1[0]);
    assert ((2, 5) == z1[1]);
    assert ((3, 6) == z1[2]);

    let (left, right) = vec::unzip(z1);

    assert ((1, 4) == (left[0], right[0]));
    assert ((2, 5) == (left[1], right[1]));
    assert ((3, 6) == (left[2], right[2]));
}

#[test]
fn test_position() {
    let v1: [int] = [1, 2, 3, 3, 2, 5];
    assert (position(1, v1) == option::some::<uint>(0u));
    assert (position(2, v1) == option::some::<uint>(1u));
    assert (position(5, v1) == option::some::<uint>(5u));
    assert (position(4, v1) == option::none::<uint>);
}

#[test]
fn test_position_pred() {
    fn less_than_three(i: &int) -> bool { ret i < 3; }
    fn is_eighteen(i: &int) -> bool { ret i == 18; }
    let v1: [int] = [5, 4, 3, 2, 1];
    assert (position_pred(less_than_three, v1) == option::some::<uint>(3u));
    assert (position_pred(is_eighteen, v1) == option::none::<uint>);
}

#[test]
fn reverse_and_reversed() {
    let v: [mutable int] = [mutable 10, 20];
    assert (v[0] == 10);
    assert (v[1] == 20);
    vec::reverse(v);
    assert (v[0] == 20);
    assert (v[1] == 10);
    let v2 = vec::reversed::<int>([10, 20]);
    assert (v2[0] == 20);
    assert (v2[1] == 10);
    v[0] = 30;
    assert (v2[0] == 20);
    // Make sure they work with 0-length vectors too.

    let v4 = vec::reversed::<int>([]);
    assert v4 == [];
    let v3: [mutable int] = [mutable];
    vec::reverse::<int>(v3);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

