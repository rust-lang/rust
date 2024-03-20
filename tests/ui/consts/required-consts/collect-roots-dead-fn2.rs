//@revisions: noopt opt
//@ build-pass
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O

//! A slight variant of `collect-roots-in-dead-fn` where the dead call is itself generic. Now this
//! *passes* in both optimized and unoptimized builds: the call to `h` always disappears in an early
//! `SimplifyCfg`, and `h` is generic so it can never be a root.
#![crate_type = "lib"]

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!();
}

pub fn f() {
    loop {}; h::<i32>()
}

#[inline(never)]
fn h<T>() {
    Fail::<T>::C;
}
