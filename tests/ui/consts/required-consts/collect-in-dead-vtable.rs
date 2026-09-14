//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation panicked: explicit panic
}

trait MyTrait {
    fn not_called(&self);
}

// This function is not actually called, but it is mentioned in a vtable in a function that is
// called. Make sure we still find this error.
// This ensures that we are properly considering vtables when gathering "mentioned" items.
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
        let gen_vtable: &dyn MyTrait = &v; // vtable is "mentioned" here
    }
}

pub fn main() {
    called::<i32>();
}
