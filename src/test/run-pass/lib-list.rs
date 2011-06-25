
use std;
import std::list;
import std::list::car;
import std::list::cdr;
import std::list::from_vec;
import std::option;

fn test_from_vec() {
    auto l = from_vec([0, 1, 2]);
    assert (car(l) == 0);
    assert (car(cdr(l)) == 1);
    assert (car(cdr(cdr(l))) == 2);
}

fn test_foldl() {
    auto l = from_vec([0, 1, 2, 3, 4]);
    fn add(&int a, &uint b) -> uint { ret (a as uint) + b; }
    auto rs = list::foldl(l, 0u, add);
    assert (rs == 10u);
}

fn test_find_success() {
    auto l = from_vec([0, 1, 2]);
    fn match(&int i) -> option::t[int] {
        ret if (i == 2) { option::some(i) } else { option::none[int] };
    }
    auto rs = list::find(l, match);
    assert (rs == option::some(2));
}

fn test_find_fail() {
    auto l = from_vec([0, 1, 2]);
    fn match(&int i) -> option::t[int] { ret option::none[int]; }
    auto rs = list::find(l, match);
    assert (rs == option::none[int]);
}

fn test_has() {
    auto l = from_vec([5, 8, 6]);
    auto empty = list::nil[int];
    assert (list::has(l, 5));
    assert (!list::has(l, 7));
    assert (list::has(l, 8));
    assert (!list::has(empty, 5));
}

fn test_length() {
    auto l = from_vec([0, 1, 2]);
    assert (list::length(l) == 3u);
}

fn main() {
    test_from_vec();
    test_foldl();
    test_find_success();
    test_find_fail();
    test_length();
    test_has();
}