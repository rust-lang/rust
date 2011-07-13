

// -*- rust -*-
use std;
import std::str;

#[test]
fn test() {
    auto s = "hello";
    auto sb = str::buf(s);
    auto s_cstr = str::str_from_cstr(sb);
    assert (str::eq(s_cstr, s));
    auto s_buf = str::str_from_buf(sb, 5u);
    assert (str::eq(s_buf, s));
}