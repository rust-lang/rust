#[repr(packed)]
struct S {
    a: i32,
    b: i64,
}

fn main() {
    let x = S {
        a: 42,
        b: 99,
    };
    let a = x.a;
    let b = x.b;
    assert_eq!(a, 42);
    assert_eq!(b, 99);
    // can't do `assert_eq!(x.a, 42)`, because `assert_eq!` takes a reference
    assert_eq!({x.a}, 42);
    assert_eq!({x.b}, 99);
}
