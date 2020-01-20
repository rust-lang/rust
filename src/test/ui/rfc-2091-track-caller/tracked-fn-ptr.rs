// run-pass

#![feature(track_caller)]

fn ptr_call(f: fn()) {
    f();
}

#[track_caller]
fn tracked() {
    let expected_line = line!() - 1;
    let location = std::panic::Location::caller();
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
}

fn main() {
    ptr_call(tracked);
}
