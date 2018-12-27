// run-pass
#[derive(PartialEq, Hash, Debug)]
struct Foo<T> {
    x: isize,
    y: T,
    z: isize
}

pub fn main() {
    let a = Foo { x: 1, y: 2.0f64, z: 3 };
    let b = Foo { x: 1, y: 2.0f64, z: 3 };
    assert_eq!(a, b);
    assert!(!(a != b));
    assert!(a.eq(&b));
    assert!(!a.ne(&b));
}
