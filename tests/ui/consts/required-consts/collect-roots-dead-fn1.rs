//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O

//! This used to fail in optimized builds but pass in unoptimized builds. The reason is that in
//! optimized builds, `f` gets marked as cross-crate-inlineable, so the functions it calls become
//! reachable, and therefore `g` becomes a collection root. But in unoptimized builds, `g` is no
//! root, and the call to `g` disappears in an early `SimplifyCfg` before "mentioned items" are
//! gathered, so we never reach `g`.
#![crate_type = "lib"]

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR: evaluation of `Fail::<i32>::C` failed
}

pub fn f() {
    loop {}; g()
}

#[inline(never)]
fn g() {
    h::<i32>()
}

// Make sure we only use the faulty const in a generic function, or
// else it gets evaluated by some MIR pass.
#[inline(never)]
fn h<T>() {
    Fail::<T>::C;
}
