import core::*;

import vec;
import vec::*;
import option;
import option::none;
import option::some;
import task;


fn square(n: uint) -> uint { ret n * n; }

fn square_ref(&&n: uint) -> uint { ret n * n; }

pure fn is_three(&&n: uint) -> bool { ret n == 3u; }

pure fn is_odd(&&n: uint) -> bool { ret n % 2u == 1u; }

pure fn is_equal(&&x: uint, &&y:uint) -> bool { ret x == y; }

fn square_if_odd(&&n: uint) -> option::t<uint> {
    ret if n % 2u == 1u { some(n * n) } else { none };
}

fn add(&&x: uint, &&y: uint) -> uint { ret x + y; }

#[test]
fn test_unsafe_ptrs() unsafe {
    // Test on-stack copy-from-buf.
    let a = [1, 2, 3];
    let ptr = vec::to_ptr(a);
    let b = vec::unsafe::from_buf(ptr, 3u);
    assert (vec::len(b) == 3u);
    assert (b[0] == 1);
    assert (b[1] == 2);
    assert (b[2] == 3);

    // Test on-heap copy-from-buf.
    let c = [1, 2, 3, 4, 5];
    ptr = vec::to_ptr(c);
    let d = vec::unsafe::from_buf(ptr, 5u);
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
fn test_push() {
    // Test on-stack push().
    let v = [];
    vec::push(v, 1);
    assert (vec::len(v) == 1u);
    assert (v[0] == 1);

    // Test on-heap push().
    vec::push(v, 2);
    assert (vec::len(v) == 2u);
    assert (v[0] == 1);
    assert (v[1] == 2);
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
    let w = vec::map(v, square_ref);
    assert (vec::len(w) == 3u);
    assert (w[0] == 1u);
    assert (w[1] == 4u);
    assert (w[2] == 9u);

    // Test on-heap map.
    v = [1u, 2u, 3u, 4u, 5u];
    w = vec::map(v, square_ref);
    assert (vec::len(w) == 5u);
    assert (w[0] == 1u);
    assert (w[1] == 4u);
    assert (w[2] == 9u);
    assert (w[3] == 16u);
    assert (w[4] == 25u);
}

#[test]
fn test_map2() {
    fn times(&&x: int, &&y: int) -> int { ret x * y; }
    let f = times;
    let v0 = [1, 2, 3, 4, 5];
    let v1 = [5, 4, 3, 2, 1];
    let u = vec::map2::<int, int, int>(v0, v1, f);
    let i = 0;
    while i < 5 { assert (v0[i] * v1[i] == u[i]); i += 1; }
}

#[test]
fn test_filter_map() {
    // Test on-stack filter-map.
    let v = [1u, 2u, 3u];
    let w = vec::filter_map(v, square_if_odd);
    assert (vec::len(w) == 2u);
    assert (w[0] == 1u);
    assert (w[1] == 9u);

    // Test on-heap filter-map.
    v = [1u, 2u, 3u, 4u, 5u];
    w = vec::filter_map(v, square_if_odd);
    assert (vec::len(w) == 3u);
    assert (w[0] == 1u);
    assert (w[1] == 9u);
    assert (w[2] == 25u);

    fn halve(&&i: int) -> option::t<int> {
        if i % 2 == 0 {
            ret option::some::<int>(i / 2);
        } else { ret option::none::<int>; }
    }
    fn halve_for_sure(&&i: int) -> int { ret i / 2; }
    let all_even: [int] = [0, 2, 8, 6];
    let all_odd1: [int] = [1, 7, 3];
    let all_odd2: [int] = [];
    let mix: [int] = [9, 2, 6, 7, 1, 0, 0, 3];
    let mix_dest: [int] = [1, 3, 0, 0];
    assert (filter_map(all_even, halve) == map(all_even, halve_for_sure));
    assert (filter_map(all_odd1, halve) == []);
    assert (filter_map(all_odd2, halve) == []);
    assert (filter_map(mix, halve) == mix_dest);
}

#[test]
fn test_filter() {
    assert filter([1u, 2u, 3u], is_odd) == [1u, 3u];
    assert filter([1u, 2u, 4u, 8u, 16u], is_three) == [];
}

#[test]
fn test_foldl() {
    // Test on-stack fold.
    let v = [1u, 2u, 3u];
    let sum = vec::foldl(0u, v, add);
    assert (sum == 6u);

    // Test on-heap fold.
    v = [1u, 2u, 3u, 4u, 5u];
    sum = vec::foldl(0u, v, add);
    assert (sum == 15u);
}

#[test]
fn test_foldl2() {
    fn sub(&&a: int, &&b: int) -> int {
        a - b
    }
    let v = [1, 2, 3, 4];
    let sum = vec::foldl(0, v, sub);
    assert sum == -10;
}

#[test]
fn test_foldr() {
    fn sub(&&a: int, &&b: int) -> int {
        a - b
    }
    let v = [1, 2, 3, 4];
    let sum = vec::foldr(v, 0, sub);
    assert sum == -2;
}

#[test]
fn iter_empty() {
    let i = 0;
    vec::iter::<int>([], { |_v| i += 1 });
    assert i == 0;
}

#[test]
fn iter_nonempty() {
    let i = 0;
    vec::iter([1, 2, 3], { |v| i += v });
    assert i == 6;
}

#[test]
fn iteri() {
    let i = 0;
    vec::iteri([1, 2, 3], { |j, v|
        if i == 0 { assert v == 1; }
        assert j + 1u == v as uint;
        i += v;
                          });
    assert i == 6;
}

#[test]
fn riter_empty() {
    let i = 0;
    vec::riter::<int>([], { |_v| i += 1 });
    assert i == 0;
}

#[test]
fn riter_nonempty() {
    let i = 0;
    vec::riter([1, 2, 3], { |v|
        if i == 0 { assert v == 3; }
        i += v
                         });
    assert i == 6;
}

#[test]
fn riteri() {
    let i = 0;
    vec::riteri([0, 1, 2], { |j, v|
        if i == 0 { assert v == 2; }
        assert j == v as uint;
        i += v;
                          });
    assert i == 3;
}

#[test]
fn test_permute() {
  let results: [[int]];

  results = [];
  permute([]) {|v| results += [v]; }
  assert results == [[]];

  results = [];
  permute([7]) {|v| results += [v]; }
  assert results == [[7]];

  results = [];
  permute([1,1]) {|v| results += [v]; }
  assert results == [[1,1],[1,1]];

  results = [];
  permute([5,2,0]) {|v| results += [v]; }
  assert results == [[5,2,0],[5,0,2],[2,5,0],[2,0,5],[0,5,2],[0,2,5]];
}

#[test]
fn test_any_and_all() {
    assert (vec::any([1u, 2u, 3u], is_three));
    assert (!vec::any([0u, 1u, 2u], is_three));
    assert (vec::any([1u, 2u, 3u, 4u, 5u], is_three));
    assert (!vec::any([1u, 2u, 4u, 5u, 6u], is_three));

    assert (vec::all([3u, 3u, 3u], is_three));
    assert (!vec::all([3u, 3u, 2u], is_three));
    assert (vec::all([3u, 3u, 3u, 3u, 3u], is_three));
    assert (!vec::all([3u, 3u, 0u, 1u, 2u], is_three));
}

#[test]
fn test_any2_and_all2() {

    assert (vec::any2([2u, 4u, 6u], [2u, 4u, 6u], is_equal));
    assert (vec::any2([1u, 2u, 3u], [4u, 5u, 3u], is_equal));
    assert (!vec::any2([1u, 2u, 3u], [4u, 5u, 6u], is_equal));
    assert (vec::any2([2u, 4u, 6u], [2u, 4u], is_equal));

    assert (vec::all2([2u, 4u, 6u], [2u, 4u, 6u], is_equal));
    assert (!vec::all2([1u, 2u, 3u], [4u, 5u, 3u], is_equal));
    assert (!vec::all2([1u, 2u, 3u], [4u, 5u, 6u], is_equal));
    assert (!vec::all2([2u, 4u, 6u], [2u, 4u], is_equal));
}

#[test]
fn test_zip_unzip() {
    let v1 = [1, 2, 3];
    let v2 = [4, 5, 6];

    check (same_length(v1, v2)); // Silly, but what else can we do?
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
    fn less_than_three(&&i: int) -> bool { ret i < 3; }
    fn is_eighteen(&&i: int) -> bool { ret i == 18; }
    let v1: [int] = [5, 4, 3, 2, 1];
    assert (position_pred(v1, less_than_three) == option::some::<uint>(3u));
    assert (position_pred(v1, is_eighteen) == option::none::<uint>);
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
    assert (v4 == []);
    let v3: [mutable int] = [mutable];
    vec::reverse::<int>(v3);
}

#[test]
fn reversed_mut() {
    let v2 = vec::reversed::<int>([mutable 10, 20]);
    assert (v2[0] == 20);
    assert (v2[1] == 10);
}

#[test]
fn init() {
    let v = vec::init([1, 2, 3]);
    assert v == [1, 2];
}

#[test]
// FIXME: Windows can't undwind
#[ignore(cfg(target_os = "win32"))]
fn init_empty() {

    let r = task::join(
        task::spawn_joinable {||
            task::unsupervise();
            vec::init::<int>([]);
        });
    assert r == task::tr_failure
}

#[test]
fn concat() {
    assert vec::concat([[1], [2,3]]) == [1, 2, 3];
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

