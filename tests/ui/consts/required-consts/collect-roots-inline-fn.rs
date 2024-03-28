//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O

//! In optimized builds, the functions in this crate are all marked "inline" so none of them become
//! collector roots. Ensure that we still evaluate the constants.
#![crate_type = "lib"]

struct Zst<T>(T);
impl<T> Zst<T> {
    const ASSERT: () = if std::mem::size_of::<T>() != 0 {
        panic!(); //~ERROR: evaluation of `Zst::<u32>::ASSERT` failed
    };
}

fn f<T>() {
    Zst::<T>::ASSERT;
}

pub fn g() {
    f::<u32>()
}
