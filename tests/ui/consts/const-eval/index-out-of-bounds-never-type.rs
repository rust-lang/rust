//@ build-fail
//@ ignore-parallel-frontend post-monomorphization errors
// Regression test for #66975
#![warn(unconditional_panic)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = { let x = 0 * std::mem::size_of::<T>(); [][x] };
    //~^ ERROR index out of bounds: the length is 0 but the index is 0

}

fn f<T>() {
    let _ = PrintName::<T>::VOID;
}

pub fn main() {
    f::<()>();
}
