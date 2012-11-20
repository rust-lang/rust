#[deriving_eq]
struct Foo {
    x: int,
    y: int,
    z: int,
}

fn main() {
    let a = Foo { x: 1, y: 2, z: 3 };
    let b = Foo { x: 1, y: 2, z: 3 };
    assert a == b;
    assert !(a != b);
    assert a.eq(&b);
    assert !a.ne(&b);
}

