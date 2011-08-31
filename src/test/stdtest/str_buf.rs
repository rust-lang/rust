

// -*- rust -*-
use std;
import std::istr;

#[test]
fn test() {
    let s = ~"hello";
    let sb = istr::as_buf(s, { |b| b });
    let s_cstr = istr::str_from_cstr(sb);
    assert (istr::eq(s_cstr, s));
}
