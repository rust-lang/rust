type an_int = int;

fn cmp(x: option<an_int>, y: option<int>) -> bool {
    x == y
}

fn main() {
    assert !cmp(some(3), none);
    assert !cmp(some(3), some(4));
    assert cmp(some(3), some(3));
    assert cmp(none, none);
}