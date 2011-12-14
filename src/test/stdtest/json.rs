import core::*;

use std;
import option;
import std::json::*;
import option::{none, some};

#[test]
fn test_from_str_num() {
    assert(from_str("3") == some(num(3f)));
    assert(from_str("3.1") == some(num(3.1f)));
    assert(from_str("-1.2") == some(num(-1.2f)));
    assert(from_str(".4") == some(num(0.4f)));
}

#[test]
fn test_from_str_str() {
    assert(from_str("\"foo\"") == some(string("foo")));
    assert(from_str("\"\\\"\"") == some(string("\"")));
    assert(from_str("\"lol") == none);
}

#[test]
fn test_from_str_bool() {
    assert(from_str("true") == some(boolean(true)));
    assert(from_str("false") == some(boolean(false)));
    assert(from_str("truz") == none);
}

#[test]
fn test_from_str_list() {
    assert(from_str("[]") == some(list(@[])));
    assert(from_str("[true]") == some(list(@[boolean(true)])));
    assert(from_str("[3, 1]") == some(list(@[num(3f), num(1f)])));
    assert(from_str("[2, [4, 1]]") ==
        some(list(@[num(2f), list(@[num(4f), num(1f)])])));
    assert(from_str("[2, ]") == none);
    assert(from_str("[5, ") == none);
    assert(from_str("[6 7]") == none);
    assert(from_str("[3") == none);
}

#[test]
fn test_from_str_dict() {
    assert(from_str("{}") != none);
    assert(from_str("{\"a\": 3}") != none);
    assert(from_str("{\"a\": }") == none);
    assert(from_str("{\"a\" }") == none);
    assert(from_str("{\"a\"") == none);
    assert(from_str("{") == none);
}
