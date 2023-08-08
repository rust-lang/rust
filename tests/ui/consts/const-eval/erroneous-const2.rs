//! Make sure we error on erroneous consts even if they are unused.
#![allow(unconditional_panic)]

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = [()][2]; //~ERROR evaluation of `PrintName::<i32>::VOID` failed
}

pub static FOO: () = {
    if false {
        // This bad constant is only used in dead code in a static initializer... and yet we still
        // must make sure that the build fails.
        PrintName::<i32>::VOID; //~ constant
    }
};

fn main() {
    FOO
}
