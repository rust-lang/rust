use std;
import std::vec;

fn main() {
    // Make sure we properly handle repeated self-appends.
    let a: vec[int] = [0];
    let i = 20;
    let expected_len = 1u;
    while i > 0 {
        log_err vec::len(a);
        assert (vec::len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}