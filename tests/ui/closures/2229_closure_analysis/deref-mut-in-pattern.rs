// Newly accepted examples as a result of the changes introduced in #138961.
//
//@ edition:2024
//@ check-pass
#![allow(unused_assignments)]

// Reading the length as part of a pattern captures the pointee.
fn f() {
    let mut x: &mut [u8] = &mut [1, 2, 3];
    let c = || {
        match x {
            [] => (),
            _ => (),
        }
    };
    x = &mut [];
    c();
}

// Plain old deref as part of pattern behaves similarly
fn g() {
    let mut x: &mut bool = &mut false;
    let mut t = true;
    let c = || {
        match x {
            true => (),
            false => (),
        }
    };
    x = &mut t;
    c();
}

// Like f, but the lifetime implications are expressed in terms of
// returning a closure.
fn f2<'l: 's, 's>(x: &'s mut &'l [u8]) -> impl Fn() + 'l {
    || match *x {
        &[] => (),
        _ => (),
    }
}

// Related testcase that was already accepted before
fn f3<'l: 's, 's>(x: &'s mut &'l [u8]) -> impl Fn() + 'l {
    || match **x {
        [] => (),
        _ => (),
    }
}

fn main() {
    f();
    g();
}
