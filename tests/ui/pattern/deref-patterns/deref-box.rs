//@ run-pass
//! Deref patterns on boxes are lowered using built-in derefs, rather than generic `Deref::deref`
//! and `DerefMut::deref_mut`. Test that they work as expected.

#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn unbox_1<T>(b: Box<T>) -> T {
    let deref!(x) = b else { unreachable!() };
    x
}

fn unbox_2<T>(b: Box<(T,)>) -> T {
    let (x,) = b else { unreachable!() };
    x
}

fn main() {
    // test that deref patterns can move out of boxes
    let b1 = Box::new(0);
    let b2 = Box::new((0,));
    assert_eq!(unbox_1(b1), unbox_2(b2));

    // test that borrowing from a box also works
    let mut b = "hi".to_owned().into_boxed_str();
    let deref!(ref mut s) = b else { unreachable!() };
    s.make_ascii_uppercase();
    assert_eq!(&*b, "HI");
}
