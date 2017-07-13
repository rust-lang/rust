#![allow(dead_code)]
#[repr(packed)]
struct S {
    a: i32,
    b: i64,
}

#[repr(packed)]
struct Test1<'a> {
    x: u8,
    other: &'a u32,
}

#[repr(packed)]
struct Test2<'a> {
    x: u8,
    other: &'a Test1<'a>,
}

fn test(t: Test2) {
    let x = *t.other.other;
    assert_eq!(x, 42);
}

fn main() {
    let mut x = S {
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

    x.b = 77;
    assert_eq!({x.b}, 77);

    test(Test2 { x: 0, other: &Test1 { x: 0, other: &42 }});
}
