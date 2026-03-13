//! A collection of very old tests of basic `Box` functionality.
//@ run-pass

fn deref_mut() {
    let mut i: Box<_> = Box::new(0);
    *i = 1;
    assert_eq!(*i, 1);
}

// Tests for if as expressions returning boxed types
fn box_if() {
    let rs: Box<_> = if true { Box::new(100) } else { Box::new(101) };
    assert_eq!(*rs, 100);
}

fn cmp() {
    let i: Box<_> = Box::new(100);
    assert_eq!(i, Box::new(100));
    assert!(i < Box::new(101));
    assert!(i <= Box::new(100));
    assert!(i > Box::new(99));
    assert!(i >= Box::new(99));
}

fn autoderef_field() {
    struct J {
        j: isize,
    }

    let i: Box<_> = Box::new(J { j: 100 });
    assert_eq!(i.j, 100);
}

fn assign_copy() {
    let mut i: Box<_> = Box::new(1);
    // Should be a copy
    let mut j;
    j = i.clone();
    *i = 2;
    *j = 3;
    assert_eq!(*i, 2);
    assert_eq!(*j, 3);
}

fn arg_mut() {
    fn f(i: &mut Box<isize>) {
        *i = Box::new(200);
    }
    let mut i = Box::new(100);
    f(&mut i);
    assert_eq!(*i, 200);
}

fn assign_generic() {
    fn f<T>(t: T) -> T {
        let t1 = t;
        t1
    }

    let t = f::<Box<_>>(Box::new(100));
    assert_eq!(t, Box::new(100));
}

pub fn main() {
    deref_mut();
    box_if();
    cmp();
    autoderef_field();
    assign_copy();
    arg_mut();
    assign_generic();
}
