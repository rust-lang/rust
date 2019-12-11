// Regression test for #66975
#![warn(const_err)]
#![feature(never_type)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = { let x = 0 * std::mem::size_of::<T>(); [][x] };
    //~^ WARN any use of this value will cause an error
}

fn f<T>() {
    let _ = PrintName::<T>::VOID;
    //~^ ERROR erroneous constant encountered
}

pub fn main() {
    f::<()>();
}
