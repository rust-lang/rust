use std;
import std::treemap::*;
import std::option::some;
import std::option::none;
import std::str;

#[test]
fn init_treemap() { let m = init::<int, int>(); }

#[test]
fn insert_one() { let m = init(); insert(m, 1, 2); }

#[test]
fn insert_two() { let m = init(); insert(m, 1, 2); insert(m, 3, 4); }

#[test]
fn insert_find() {
    let m = init();
    insert(m, 1, 2);
    assert (find(m, 1) == some(2));
}

#[test]
fn find_empty() { let m = init::<int, int>(); assert (find(m, 1) == none); }

#[test]
fn find_not_found() {
    let m = init();
    insert(m, 1, 2);
    assert (find(m, 2) == none);
}

#[test]
fn traverse_in_order() {
    let m = init();
    insert(m, 3, ());
    insert(m, 0, ());
    insert(m, 4, ());
    insert(m, 2, ());
    insert(m, 1, ());

    let n = 0;
    fn t(n: &mutable int, k: &int, v: &()) { assert (n == k); n += 1; }
    traverse(m, bind t(n, _, _));
}

#[test]
fn u8_map() {
    let m = init();

    let k1 = str::bytes("foo");
    let k2 = str::bytes("bar");

    insert(m, k1, "foo");
    insert(m, k2, "bar");

    assert (find(m, k2) == some("bar"));
    assert (find(m, k1) == some("foo"));
}
