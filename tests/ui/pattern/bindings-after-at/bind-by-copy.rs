//@ run-pass
#![allow(unused)]

// Test copy

struct A {
    a: i32,
    b: i32,
}
struct B {
    a: i32,
    b: C,
}
struct D {
    a: i32,
    d: C,
}
#[derive(Copy, Clone)]
struct C {
    c: i32,
}
enum E {
    E { a: i32, e: C },
    NotE,
}

#[rustfmt::skip]
pub fn main() {
    match (A {a: 10, b: 20}) {
        x@A {a, b: 20} => { assert!(x.a == 10); assert!(a == 10); }
        A {b: _b, ..} => { panic!(); }
    }

    let mut x@B {b, ..} = B {a: 10, b: C {c: 20}};
    assert_eq!(x.a, 10);
    x.b.c = 30;
    assert_eq!(b.c, 20);
    let mut y@D {d, ..} = D {a: 10, d: C {c: 20}};
    assert_eq!(y.a, 10);
    y.d.c = 30;
    assert_eq!(d.c, 20);

    match (E::E { a: 10, e: C { c: 20 } }) {
        x @ E::E{ a, e: C { c } } => {
            assert!(matches!(x, E::E { a: 10, e: C { c: 20 } }));
            assert!(a == 10);
            assert!(c == 20);
        }
        _ => panic!(),
    }
    match (E::E { a: 10, e: C { c: 20 } }) {
        mut x @ E::E{ a, e: C { mut c } } => {
            x = E::NotE;
            c += 30;
            assert_eq!(c, 50);
        }
        _ => panic!(),
    }

    let some_b = Some(B { a: 10, b: C { c: 20 } });

    // in irrefutable pattern
    if let Some(x @ B { b, .. }) = some_b {
        assert_eq!(x.b.c, 20);
        assert_eq!(b.c, 20);
    } else {
        unreachable!();
    }

    let some_b = Some(B { a: 10, b: C { c: 20 } });

    if let Some(x @ B { b: mut b @ C { c }, .. }) = some_b {
        assert_eq!(x.b.c, 20);
        assert_eq!(b.c, 20);
        b.c = 30;
        assert_eq!(b.c, 30);
        assert_eq!(c, 20);
    } else {
        unreachable!();
    }
}
