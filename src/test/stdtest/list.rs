
use std;
import std::list;
import std::list::car;
import std::list::cdr;
import std::list::from_vec;
import std::option;

#[test]
fn test_from_vec() {
    let l = from_vec([0, 1, 2]);
    assert (car(l) == 0);
    assert (car(cdr(l)) == 1);
    assert (car(cdr(cdr(l))) == 2);
}

#[test]
fn test_foldl() {
    let l = from_vec([0, 1, 2, 3, 4]);
    fn add(&&a: int, &&b: uint) -> uint { ret (a as uint) + b; }
    let rs = list::foldl(l, 0u, add);
    assert (rs == 10u);
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
fn test_length() {
    let l = from_vec([0, 1, 2]);
    assert (list::length(l) == 3u);
}

