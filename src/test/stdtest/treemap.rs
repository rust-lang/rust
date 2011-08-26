use std;
import std::treemap::*;
import std::option::some;
import std::option::none;

#[test]
fn init_treemap() {
    let m = init::<int, int>();
}

#[test]
fn insert_one() {
    let m = init();
    insert(m, 1, 2);
}

#[test]
fn insert_two() {
    let m = init();
    insert(m, 1, 2);
    insert(m, 3, 4);
}

#[test]
fn insert_find() {
    let m = init();
    insert(m, 1, 2);
    assert(find(m, 1) == some(2));
}

#[test]
fn find_empty() {
    let m = init::<int, int>();
    assert(find(m, 1) == none);
}

#[test]
fn find_not_found() {
    let m = init();
    insert(m, 1, 2);
    assert(find(m, 2) == none);
}
