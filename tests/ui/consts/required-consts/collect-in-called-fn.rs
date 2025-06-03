//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! Make sure we detect erroneous constants post-monomorphization even when they are unused. This is
//! crucial, people rely on it for soundness. (https://github.com/rust-lang/rust/issues/112090)

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
}

#[inline(never)]
fn called<T>() {
    // Any function that is called is guaranteed to have all consts that syntactically
    // appear in its body evaluated, even if they only appear in dead code.
    // This relies on mono-item collection checking `required_consts` in collected functions.
    if false {
        let _ = Fail::<T>::C;
    }
}

pub fn main() {
    called::<i32>();
}
