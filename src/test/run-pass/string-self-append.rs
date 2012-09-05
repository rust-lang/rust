use std;

fn main() {
    // Make sure we properly handle repeated self-appends.
    let mut a: ~str = ~"A";
    let mut i = 20;
    let mut expected_len = 1u;
    while i > 0 {
        log(error, str::len(a));
        assert (str::len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}
