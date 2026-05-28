//@revisions: noopt opt
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! Make sure we error on erroneous consts even if they are unused.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR explicit panic
                            //~| NOTE in this expansion of panic!
}

pub static FOO: () = {
    if false {
        // This bad constant is only used in dead code in a static initializer... and yet we still
        // must make sure that the build fails.
        // This relies on const-eval evaluating all `required_consts` of the `static` MIR body.
        Fail::<i32>::C; //~ NOTE constant
    }
};

fn main() {
    FOO
}
