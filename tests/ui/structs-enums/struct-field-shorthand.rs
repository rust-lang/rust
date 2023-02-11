// run-pass
struct Foo {
    x: i32,
    y: bool,
    z: i32
}

struct Bar {
    x: i32
}

pub fn main() {
    let (x, y, z) = (1, true, 2);
    let a = Foo { x, y: y, z };
    assert_eq!(a.x, x);
    assert_eq!(a.y, y);
    assert_eq!(a.z, z);

    let b = Bar { x, };
    assert_eq!(b.x, x);

    let c = Foo { z, y, x };
    assert_eq!(c.x, x);
    assert_eq!(c.y, y);
    assert_eq!(c.z, z);
}
