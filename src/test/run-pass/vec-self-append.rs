// xfail-stage1
// xfail-stage2
// xfail-stage3

use std;
import std::ivec;

fn main() {
    // Make sure we properly handle repeated self-appends.
    let a: [int] = ~[0];
    let i = 20;
    let expected_len = 1u;
    while i > 0 {
        log_err ivec::len(a);
        assert (ivec::len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}