// run-pass

#![feature(track_caller)]

fn ptr_call(f: fn()) {
    f();
}

#[track_caller]
fn tracked() {
    assert_eq!(std::panic::Location::caller().file(), file!());
}

fn main() {
    ptr_call(tracked);
}
