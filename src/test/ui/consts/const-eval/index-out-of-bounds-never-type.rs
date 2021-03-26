// build-fail

// Regression test for #66975
#![warn(const_err, unconditional_panic)]
#![feature(never_type)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = { let x = 0 * std::mem::size_of::<T>(); [][x] };
    //~^ WARN any use of this value will cause an error
    //~| WARN this was previously accepted by the compiler but is being phased out

}

fn f<T>() {
    let _ = PrintName::<T>::VOID;
    //~^ ERROR erroneous constant encountered
}

pub fn main() {
    f::<()>();
}
