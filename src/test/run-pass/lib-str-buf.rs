// -*- rust -*-

use std;
import std._str;

fn main() {
    auto s = "hello";
    auto sb = _str.rustrt.str_buf(s);
    auto s_cstr = _str.rustrt.str_from_cstr(sb);
    check (_str.eq(s_cstr, s));
    auto s_buf = _str.rustrt.str_from_buf(sb, 5u);
    check (_str.eq(s_buf, s));
}

