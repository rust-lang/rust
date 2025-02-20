//@revisions: noopt opt
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! Make sure we evaluate const fn calls even if they get promoted and their result ignored.

const unsafe fn ub() {
    std::hint::unreachable_unchecked(); //~ inside `ub`
}

pub const FOO: () = unsafe {
    // Make sure that this gets promoted and then fails to evaluate, and we deal with that
    // correctly.
    let _x: &'static () = &ub(); //~ ERROR evaluation of constant value failed
};

fn main() {}
