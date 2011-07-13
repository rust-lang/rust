

// -*- rust -*-
use std;
import std::str;
import std::vec;

#[test]
fn test_simple() {
    let str s1 = "All mimsy were the borogoves";
    /*
     * FIXME from_bytes(vec[u8] v) has constraint is_utf(v), which is
     * unimplemented and thereby just fails.  This doesn't stop us from
     * using from_bytes for now since the constraint system isn't fully
     * working, but we should implement is_utf8 before that happens.
     */

    let vec[u8] v = str::bytes(s1);
    let str s2 = str::from_bytes(v);
    let uint i = 0u;
    let uint n1 = str::byte_len(s1);
    let uint n2 = vec::len[u8](v);
    assert (n1 == n2);
    while (i < n1) {
        let u8 a = s1.(i);
        let u8 b = s2.(i);
        log a;
        log b;
        assert (a == b);
        i += 1u;
    }
    log "refcnt is";
    log str::refcount(s1);
}
