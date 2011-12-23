use std;
import str;

fn main() {
    // Make sure we properly handle repeated self-appends.
    let a: str = "A";
    let i = 20;
    let expected_len = 1u;
    while i > 0 {
        log(error, str::byte_len(a));
        assert (str::byte_len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}
