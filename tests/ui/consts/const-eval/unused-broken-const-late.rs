// build-fail
// compile-flags: -O
//! Make sure we detect erroneous constants post-monomorphization even when they are unused. This is
//! crucial, people rely on it for soundness. (https://github.com/rust-lang/rust/issues/112090)

struct PrintName<T>(T);
impl<T> PrintName<T> {
    const VOID: () = panic!(); //~ERROR evaluation of `PrintName::<i32>::VOID` failed
}

fn no_codegen<T>() {
    // Any function that is called is guaranteed to have all consts that syntactically
    // appear in its body evaluated, even if they only appear in dead code.
    if false {
        let _ = PrintName::<T>::VOID;
    }
}
pub fn main() {
    no_codegen::<i32>();
}
