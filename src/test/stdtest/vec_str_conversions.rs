

// -*- rust -*-
use std;
import std::istr;
import std::vec;

#[test]
fn test_simple() {
    let s1: istr = ~"All mimsy were the borogoves";

    let v: [u8] = istr::bytes(s1);
    let s2: istr = istr::unsafe_from_bytes(v);
    let i: uint = 0u;
    let n1: uint = istr::byte_len(s1);
    let n2: uint = vec::len::<u8>(v);
    assert (n1 == n2);
    while i < n1 {
        let a: u8 = s1[i];
        let b: u8 = s2[i];
        log a;
        log b;
        assert (a == b);
        i += 1u;
    }
}
