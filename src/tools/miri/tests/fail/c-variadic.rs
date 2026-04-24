//@error-in-other-file: Undefined Behavior: more C-variadic arguments read than were passed

fn read_too_many() {
    unsafe extern "C" fn variadic(mut ap: ...) {
        ap.next_arg::<i32>();
    }

    unsafe { variadic() };
}

fn main() {
    read_too_many();
}
