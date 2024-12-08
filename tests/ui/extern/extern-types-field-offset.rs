//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0
//@ normalize-stderr-test: "(core/src/panicking\.rs):[0-9]+:[0-9]+" -> "$1:$$LINE:$$COL"
#![feature(extern_types)]

extern "C" {
    type Opaque;
}

struct Newtype(Opaque);

struct S {
    i: i32,
    j: i32,
    a: Newtype,
}

fn main() {
    let buf = [0i32; 4];

    let x: &Newtype = unsafe { &*(&buf as *const _ as *const Newtype) };
    // Projecting to the newtype works, because it is always at offset 0.
    let field = &x.0;

    let x: &S = unsafe { &*(&buf as *const _ as *const S) };
    // Accessing sized fields is perfectly fine, even at non-zero offsets.
    let field = &x.i;
    let field = &x.j;
    // This needs to compute the field offset, but we don't know the type's alignment,
    // so this panics.
    let field = &x.a;
}
