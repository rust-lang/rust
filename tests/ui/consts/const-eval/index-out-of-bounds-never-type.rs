// build-fail

// Regression test for #66975
#![warn(unconditional_panic)]
#![feature(never_type)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = { let x = 0 * std::mem::size_of::<T>(); [][x] };
    //~^ ERROR evaluation of `PrintName::<()>::VOID` failed

}

fn f<T>() {
    let _ = PrintName::<T>::VOID;
}

pub fn main() {
    f::<()>();
}
