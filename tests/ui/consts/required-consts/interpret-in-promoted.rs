//@revisions: noopt opt
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//@ dont-require-annotations: NOTE

//! Make sure we evaluate const fn calls even if they get promoted and their result ignored.

const unsafe fn ub() {
    std::hint::unreachable_unchecked(); //~ NOTE inside `ub`
}

pub const FOO: () = unsafe {
    // Make sure that this gets promoted and then fails to evaluate, and we deal with that
    // correctly.
    let _x: &'static () = &ub(); //~ ERROR unreachable code
};

fn main() {}
