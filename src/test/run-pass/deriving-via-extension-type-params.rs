#[deriving_eq]
#[deriving_iter_bytes]
struct Foo<T> {
    x: int,
    y: T,
    z: int
}

fn main() {
    let a = Foo { x: 1, y: 2.0, z: 3 };
    let b = Foo { x: 1, y: 2.0, z: 3 };
    assert a == b;
    assert !(a != b);
    assert a.eq(&b);
    assert !a.ne(&b);
}

