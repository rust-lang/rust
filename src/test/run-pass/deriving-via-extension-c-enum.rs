#[deriving_eq]
enum Foo {
    Bar,
    Baz,
    Boo
}

fn main() {
    let a = Bar;
    let b = Bar;
    assert a == b;
    assert !(a != b);
    assert a.eq(&b);
    assert !a.ne(&b);
}

