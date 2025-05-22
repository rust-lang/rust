//@ run-pass
//! Deref patterns on boxes are lowered using built-in derefs, rather than generic `Deref::deref`
//! and `DerefMut::deref_mut`. Test that they work as expected.

#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn unbox_1<T>(b: Box<T>) -> T {
    let deref!(x) = b;
    x
}

fn unbox_2<T>(b: Box<(T,)>) -> T {
    let (x,) = b;
    x
}

fn unbox_separately<T>(b: Box<(T, T)>) -> (T, T) {
    let (x, _) = b;
    let (_, y) = b;
    (x, y)
}

fn main() {
    // test that deref patterns can move out of boxes
    let b1 = Box::new(0);
    let b2 = Box::new((0,));
    assert_eq!(unbox_1(b1), unbox_2(b2));
    let b3 = Box::new((1, 2));
    assert_eq!(unbox_separately(b3), (1, 2));

    // test that borrowing from a box also works
    let mut b = "hi".to_owned().into_boxed_str();
    let deref!(ref mut s) = b;
    s.make_ascii_uppercase();
    assert_eq!(&*b, "HI");
}
