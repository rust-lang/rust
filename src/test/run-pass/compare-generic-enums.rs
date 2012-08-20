type an_int = int;

fn cmp(x: Option<an_int>, y: Option<int>) -> bool {
    x == y
}

fn main() {
    assert !cmp(Some(3), None);
    assert !cmp(Some(3), Some(4));
    assert cmp(Some(3), Some(3));
    assert cmp(None, None);
}