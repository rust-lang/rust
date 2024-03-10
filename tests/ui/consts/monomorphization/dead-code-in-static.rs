//@revisions: opt no-opt
//@[opt] compile-flags: -O
//! Make sure we error on erroneous consts even if they are unused.
#![allow(unconditional_panic)]

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation of `Fail::<i32>::C` failed
}

pub static FOO: () = {
    if false {
        // This bad constant is only used in dead code in a static initializer... and yet we still
        // must make sure that the build fails.
        Fail::<i32>::C; //~ constant
    }
};

fn main() {
    FOO
}
