

// -*- rust -*-
use std;
import std::str;

fn test1() {
    let s: str = "hello";
    s += "world";
    log s;
    assert (s.(9) == 'd' as u8);
}

fn test2() {
    // This tests for issue #163

    let ff: str = "abc";
    let a: str = ff + "ABC" + ff;
    let b: str = "ABC" + ff + "ABC";
    log a;
    log b;
    assert (str::eq(a, "abcABCabc"));
    assert (str::eq(b, "ABCabcABC"));
}

fn main() { test1(); test2(); }