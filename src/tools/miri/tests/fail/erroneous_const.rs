//! Make sure we detect erroneous constants post-monomorphization even when they are unused.
//! (https://github.com/rust-lang/miri/issues/1382)
#![feature(never_type)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: ! = panic!(); //~ERROR: evaluation of `PrintName::<i32>::VOID` failed
}

fn no_codegen<T>() {
    if false {
        let _ = PrintName::<T>::VOID; //~NOTE: constant
    }
}
fn main() {
    no_codegen::<i32>();
}
