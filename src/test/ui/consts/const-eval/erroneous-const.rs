//! Make sure we error on erroneous consts even if they are unused.
#![warn(const_err, unconditional_panic)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = [()][2]; //~WARN any use of this value will cause an error
    //~^ WARN this operation will panic at runtime
    //~| WARN this was previously accepted by the compiler but is being phased out
}

const fn no_codegen<T>() {
    if false {
        // This bad constant is only used in dead code in a no-codegen function... and yet we still
        // must make sure that the build fails.
        let _ = PrintName::<T>::VOID; //~ERROR could not evaluate static initializer
    }
}

pub static FOO: () = no_codegen::<i32>();

fn main() {
    FOO
}
