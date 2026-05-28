#![feature(c_variadic)]

fn read_too_many() {
    unsafe extern "C" fn variadic(mut ap: ...) {
        ap.next_arg::<i32>(); //~ERROR: more C-variadic arguments read than were passed
    }

    unsafe { variadic() };
}

fn main() {
    read_too_many();
}
