fn main() {
    let i = ~100;
    let j = ~200;
    i <-> j;
    assert i == ~200;
    assert j == ~100;
}