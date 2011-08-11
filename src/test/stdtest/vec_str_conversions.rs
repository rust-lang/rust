

// -*- rust -*-
use std;
import std::str;
import std::vec;
import std::ivec;

#[test]
fn test_simple() {
    let s1: str = "All mimsy were the borogoves";
    /*
     * FIXME from_bytes(vec[u8] v) has constraint is_utf(v), which is
     * unimplemented and thereby just fails.  This doesn't stop us from
     * using from_bytes for now since the constraint system isn't fully
     * working, but we should implement is_utf8 before that happens.
     */

    let v: vec[u8] = ivec::to_vec(str::bytes(s1));
    let s2: str = str::unsafe_from_bytes(v);
    let i: uint = 0u;
    let n1: uint = str::byte_len(s1);
    let n2: uint = vec::len[u8](v);
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