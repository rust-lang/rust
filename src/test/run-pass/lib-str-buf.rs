// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std;
import std.Str;

fn main() {
    auto s = "hello";
    auto sb = Str.rustrt.str_buf(s);
    auto s_cstr = Str.rustrt.str_from_cstr(sb);
    assert (Str.eq(s_cstr, s));
    auto s_buf = Str.rustrt.str_from_buf(sb, 5u);
    assert (Str.eq(s_buf, s));
}

