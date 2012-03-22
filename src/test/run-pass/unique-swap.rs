fn main() {
    let mut i = ~100;
    let mut j = ~200;
    i <-> j;
    assert i == ~200;
    assert j == ~100;
}