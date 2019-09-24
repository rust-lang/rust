// run-pass
#![allow(dead_code)]

fn f<T>() -> bool {
    enum E<T> { V(T) }

    struct S<T>(T);

    true
}

fn main() {
    let b = f::<isize>();
    assert!(b);
}
