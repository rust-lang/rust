import at_vec::{build, from_fn, from_elem};

// Some code that could use that, then:
fn seq_range(lo: uint, hi: uint) -> @[uint] {
    do build |push| {
        for uint::range(lo, hi) |i| {
            push(i);
        }
    }
}

fn main() {
    assert seq_range(10, 15) == @[10, 11, 12, 13, 14];
    assert from_fn(5, |x| x+1) == @[1, 2, 3, 4, 5];
    assert from_elem(5, 3.14) == @[3.14, 3.14, 3.14, 3.14, 3.14];
}
