

// -*- rust -*-
use std;
import std::str;
import std::ivec;

#[test]
fn test_simple() {
    let s1: str = "All mimsy were the borogoves";

    let v: [u8] = str::bytes(s1);
    let s2: str = str::unsafe_from_bytes(v);
    let i: uint = 0u;
    let n1: uint = str::byte_len(s1);
    let n2: uint = ivec::len[u8](v);
    assert (n1 == n2);
    while i < n1 {
        let a: u8 = s1.(i);
        let b: u8 = s2.(i);
        log a;
        log b;
        assert (a == b);
        i += 1u;
    }
    log "refcnt is";
    log str::refcount(s1);
}