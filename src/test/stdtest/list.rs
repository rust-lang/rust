import core::*;

use std;
import std::list;
import std::list::{from_vec, head, is_empty, is_not_empty, tail};
import option;

#[test]
fn test_is_empty() {
    let empty : list::list<int> = from_vec([]);
    let full1 = from_vec([1]);
    let full2 = from_vec(['r', 'u']);

    assert is_empty(empty);
    assert !is_empty(full1);
    assert !is_empty(full2);

    assert !is_not_empty(empty);
    assert is_not_empty(full1);
    assert is_not_empty(full2);
}

#[test]
fn test_from_vec() {
    let l = from_vec([0, 1, 2]);

    check is_not_empty(l);
    assert (head(l) == 0);

    let tail_l = tail(l);
    check is_not_empty(tail_l);
    assert (head(tail_l) == 1);

    let tail_tail_l = tail(tail_l);
    check is_not_empty(tail_tail_l);
    assert (head(tail_tail_l) == 2);
}

#[test]
fn test_from_vec_empty() {
    let empty : list::list<int> = from_vec([]);
    assert (empty == list::nil::<int>);
}

#[test]
fn test_from_vec_mut() {
    let l = from_vec([mutable 0, 1, 2]);

    check is_not_empty(l);
    assert (head(l) == 0);

    let tail_l = tail(l);
    check is_not_empty(tail_l);
    assert (head(tail_l) == 1);

    let tail_tail_l = tail(tail_l);
    check is_not_empty(tail_tail_l);
    assert (head(tail_tail_l) == 2);
}

#[test]
fn test_foldl() {
    fn add(&&a: uint, &&b: int) -> uint { ret a + (b as uint); }
    let l = from_vec([0, 1, 2, 3, 4]);
    let empty = list::nil::<int>;
    assert (list::foldl(l, 0u, add) == 10u);
    assert (list::foldl(empty, 0u, add) == 0u);
}

#[test]
fn test_foldl2() {
    fn sub(&&a: int, &&b: int) -> int {
        a - b
    }
    let l = from_vec([1, 2, 3, 4]);
    assert (list::foldl(l, 0, sub) == -10);
}

#[test]
fn test_find_success() {
    fn match(&&i: int) -> option::t<int> {
        ret if i == 2 { option::some(i) } else { option::none::<int> };
    }
    let l = from_vec([0, 1, 2]);
    assert (list::find(l, match) == option::some(2));
}

#[test]
fn test_find_fail() {
    fn match(&&_i: int) -> option::t<int> { ret option::none::<int>; }
    let l = from_vec([0, 1, 2]);
    let empty = list::nil::<int>;
    assert (list::find(l, match) == option::none::<int>);
    assert (list::find(empty, match) == option::none::<int>);
}

#[test]
fn test_has() {
    let l = from_vec([5, 8, 6]);
    let empty = list::nil::<int>;
    assert (list::has(l, 5));
    assert (!list::has(l, 7));
    assert (list::has(l, 8));
    assert (!list::has(empty, 5));
}

#[test]
fn test_len() {
    let l = from_vec([0, 1, 2]);
    let empty = list::nil::<int>;
    assert (list::len(l) == 3u);
    assert (list::len(empty) == 0u);
}

