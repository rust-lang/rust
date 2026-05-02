//@error-in-other-file: requested `i32` is incompatible with next argument of type `()`
#![feature(c_variadic)]

// While 1-ZST are currently ignored on most ABIs, we don't guarantee that, and it's UB to
// rely on it.

fn main() {
    unsafe extern "C" fn variadic(mut ap: ...) {
        ap.next_arg::<i32>();
        ap.next_arg::<i32>(); // this one errors
    }

    unsafe { variadic(0i32, (), 1i32) }
}
