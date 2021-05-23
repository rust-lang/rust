//! Make sure we error on erroneous consts even if they are unused.
#![warn(const_err, unconditional_panic)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = [()][2]; //~WARN any use of this value will cause an error
    //~^ WARN this operation will panic at runtime
    //~| WARN this was previously accepted by the compiler but is being phased out
}

pub static FOO: () = {
    if false {
        // This bad constant is only used in dead code in a static initializer... and yet we still
        // must make sure that the build fails.
        let _ = PrintName::<i32>::VOID; //~ERROR could not evaluate static initializer
    }
};

fn main() {
    FOO
}
