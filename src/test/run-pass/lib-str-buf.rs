// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std;
import std::_str;

fn main() {
    auto s = "hello";
    auto sb = str.rustrt.str_buf(s);
    auto s_cstr = str.rustrt.str_from_cstr(sb);
    assert (str.eq(s_cstr, s));
    auto s_buf = str.rustrt.str_from_buf(sb, 5u);
    assert (str.eq(s_buf, s));
}

