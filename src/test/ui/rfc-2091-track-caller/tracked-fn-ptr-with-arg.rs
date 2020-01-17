// run-pass

#![feature(track_caller)]

fn pass_to_ptr_call<T>(f: fn(T), x: T) {
    f(x);
}

#[track_caller]
fn tracked_unit(_: ()) {
    assert_eq!(std::panic::Location::caller().file(), file!());
}

fn main() {
    pass_to_ptr_call(tracked_unit, ());
}
