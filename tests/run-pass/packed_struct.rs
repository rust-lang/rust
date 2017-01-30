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
    assert_eq!(x.a, 42);
    assert_eq!(x.b, 99);
}
