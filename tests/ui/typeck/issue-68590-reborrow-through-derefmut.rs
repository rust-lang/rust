//@ check-pass

// rust-lang/rust#68590: confusing diagnostics when reborrowing through DerefMut.

use std::cell::RefCell;

struct A;

struct S<'a> {
    a: &'a mut A,
}

fn take_a(_: &mut A) {}

fn test<'a>(s: &RefCell<S<'a>>) {
    let mut guard = s.borrow_mut();
    take_a(guard.a);
    let _s2 = S { a: guard.a };
}

fn main() {
    let a = &mut A;
    let s = RefCell::new(S { a });
    test(&s);
}
