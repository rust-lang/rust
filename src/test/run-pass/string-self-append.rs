use std;
import std::istr;

fn main() {
    // Make sure we properly handle repeated self-appends.
    let a: istr = ~"A";
    let i = 20;
    let expected_len = 1u;
    while i > 0 {
        log_err istr::byte_len(a);
        assert (istr::byte_len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}
