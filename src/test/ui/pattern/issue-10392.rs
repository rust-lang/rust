// run-pass
#![allow(unused_variables)]

struct A { foo: isize }
struct B { a: isize, b: isize, c: isize }

fn mka() -> A { panic!() }
fn mkb() -> B { panic!() }

fn test() {
    let A { foo, } = mka();
    let A {
        foo,
    } = mka();

    let B { a, b, c, } = mkb();

    match mka() {
        A { foo: _foo, } => {}
    }

    match Some(mka()) {
        Some(A { foo: _foo, }) => {}
        None => {}
    }
}

pub fn main() {
    if false { test() }
}
