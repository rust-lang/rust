use std;
import std::vec;

fn main() {
    let v = std::vec::map2({|&i, &b| if b { -i } else { i }},
                           ~[1, 2, 3, 4, 5],
                           ~[true, false, false, true, true]);
    log_err v;
    assert v == ~[-1, 2, 3, -4, -5];
}
