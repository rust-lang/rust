
use std;
import std::list;
import std::list::head;
import std::list::tail;
import std::list::from_vec;
import std::option;

#[test]
fn test_from_vec() {
    let l = from_vec([0, 1, 2]);
    assert (head(l) == 0);
    assert (head(tail(l)) == 1);
    assert (head(tail(tail(l))) == 2);
}

#[test]
fn test_from_vec_mut() {
    let l = from_vec([mutable 0, 1, 2]);
    assert (head(l) == 0);
    assert (head(tail(l)) == 1);
    assert (head(tail(tail(l))) == 2);
}

#[test]
fn test_foldl() {
    let l = from_vec([0, 1, 2, 3, 4]);
    fn add(&&a: uint, &&b: int) -> uint { ret a + (b as uint); }
    let rs = list::foldl(l, 0u, add);
    assert (rs == 10u);
}

#[test]
fn test_foldl2() {
    fn sub(&&a: int, &&b: int) -> int {
        a - b
    }
    let l = from_vec([1, 2, 3, 4]);
    let sum = list::foldl(l, 0, sub);
    assert sum == -10;
}

#[test]
fn test_find_success() {
    let l = from_vec([0, 1, 2]);
    fn match(&&i: int) -> option::t<int> {
        ret if i == 2 { option::some(i) } else { option::none::<int> };
    }
    let rs = list::find(l, match);
    assert (rs == option::some(2));
}

#[test]
fn test_find_fail() {
    let l = from_vec([0, 1, 2]);
    fn match(&&_i: int) -> option::t<int> { ret option::none::<int>; }
    let rs = list::find(l, match);
    assert (rs == option::none::<int>);
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
    assert (list::len(l) == 3u);
}

