fn main() {
    let i = ~100;
    assert i == ~100;
    assert i < ~101;
    assert i <= ~100;
    assert i > ~99;
    assert i >= ~99;
}