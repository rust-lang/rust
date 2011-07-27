

// -*- rust -*-
use std;
import std::str;

#[test]
fn test() {
    let s = "hello";
    let sb = str::buf(s);
    let s_cstr = str::str_from_cstr(sb);
    assert (str::eq(s_cstr, s));
    let s_buf = str::str_from_buf(sb, 5u);
    assert (str::eq(s_buf, s));
}