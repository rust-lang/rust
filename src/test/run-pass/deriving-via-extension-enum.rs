#[deriving_eq]
enum Foo {
    Bar(int, int),
    Baz(float, float)
}

fn main() {
    let a = Bar(1, 2);
    let b = Bar(1, 2);
    assert a == b;
    assert !(a != b);
    assert a.eq(&b);
    assert !a.ne(&b);
}

