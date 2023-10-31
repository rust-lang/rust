//! Make sure we error on erroneous consts even if they are unused.
#![allow(unconditional_panic)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = [()][2]; //~ERROR evaluation of `PrintName::<i32>::VOID` failed
}

const fn no_codegen<T>() {
    if false {
        // This bad constant is only used in dead code in a no-codegen function... and yet we still
        // must make sure that the build fails.
            PrintName::<T>::VOID; //~ constant
    }
}

pub static FOO: () = no_codegen::<i32>();

fn main() {
    FOO
}
