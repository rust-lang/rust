//@revisions: noopt opt
//@[opt] compile-flags: -O
//! Make sure we error on erroneous consts even if they are unused.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation of `Fail::<i32>::C` failed
}

#[inline(never)]
const fn no_codegen<T>() {
    if false {
        // This bad constant is only used in dead code in a no-codegen function... and yet we still
        // must make sure that the build fails.
        // This relies on const-eval evaluating all `required_consts` of `const fn`.
        Fail::<T>::C; //~ constant
    }
}

pub static FOO: () = no_codegen::<i32>();

fn main() {
    FOO
}
