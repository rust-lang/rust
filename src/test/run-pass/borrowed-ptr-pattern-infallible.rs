fn main() {
    let (&x, &y, &z) = (&3, &'a', &@"No pets!");
    assert x == 3;
    assert y == 'a';
    assert z == @"No pets!";
}
