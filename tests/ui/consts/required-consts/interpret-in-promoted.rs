//@revisions: noopt opt
//@[opt] compile-flags: -O
//! Make sure we error on erroneous consts even if they are unused.

const unsafe fn ub() {
    std::hint::unreachable_unchecked();
}

pub const FOO: () = unsafe {
    // Make sure that this gets promoted and then fails to evaluate, and we deal with that
    // correctly.
    let _x: &'static () = &ub(); //~ erroneous constant
};

fn main() {}
