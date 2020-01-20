// run-pass

#![feature(track_caller)]

fn pass_to_ptr_call<T>(f: fn(T), x: T) {
    f(x);
}

#[track_caller]
fn tracked_unit(_: ()) {
    let expected_line = line!() - 1;
    let location = std::panic::Location::caller();
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
}

fn main() {
    pass_to_ptr_call(tracked_unit, ());
}
