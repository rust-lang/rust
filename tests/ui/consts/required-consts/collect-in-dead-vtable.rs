//@revisions: noopt opt
//@[noopt] build-fail
//@[opt] compile-flags: -O
//FIXME: `opt` revision currently does not stop with an error due to
//<https://github.com/rust-lang/rust/issues/107503>.
//@[opt] build-pass
//! Make sure we detect erroneous constants post-monomorphization even when they are unused. This is
//! crucial, people rely on it for soundness. (https://github.com/rust-lang/rust/issues/112090)

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //[noopt]~ERROR evaluation of `Fail::<i32>::C` failed
}

trait MyTrait {
    fn not_called(&self);
}

// This function is not actually called, but it is mentioned in a vtable in a function that is
// called. Make sure we still find this error.
// This relies on mono-item collection checking `required_consts` in functions that are referenced
// in vtables that syntactically appear in collected functions (even inside dead code).
impl<T> MyTrait for Vec<T> {
    fn not_called(&self) {
        if false {
            let _ = Fail::<T>::C;
        }
    }
}

#[inline(never)]
fn called<T>() {
    if false {
        let v: Vec<T> = Vec::new();
        let gen_vtable: &dyn MyTrait = &v; // vtable "appears" here
    }
}

pub fn main() {
    called::<i32>();
}
