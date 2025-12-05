//@ run-fail
//@ error-pattern: panic
// for some reason, fails to match error string on
// wasm32-unknown-unknown with stripped debuginfo and symbols,
// so don't strip it
//@ compile-flags:-Cstrip=none

fn main() {
    Box::new(panic!());
}
