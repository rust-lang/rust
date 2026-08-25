//! Test that function and closure parameters marked as `mut` can be mutated
//! within the function body.

//@ run-pass

fn f(mut y: Box<isize>) {
    *y = 5;
    assert_eq!(*y, 5);
}

fn g() {
    let frob = |mut q: Box<isize>| {
        *q = 2;
        assert_eq!(*q, 2);
    };
    let w = Box::new(37);
    frob(w);
}

pub fn main() {
    let z = Box::new(17);
    f(z);
    g();
}
