//! Make sure we error on erroneous consts even if they are unused.
#![warn(const_err, unconditional_panic)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = [()][2]; //~WARN any use of this value will cause an error
    //~^ WARN this operation will panic at runtime
}

const fn no_codegen<T>() {
    if false { //~ERROR evaluation of constant value failed
        let _ = PrintName::<T>::VOID;
    }
}

pub static FOO: () = no_codegen::<i32>(); //~ERROR could not evaluate static initializer

fn main() {
    FOO
}
