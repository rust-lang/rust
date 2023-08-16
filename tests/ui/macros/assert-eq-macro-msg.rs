// run-fail
//@error-in-other-file:panicked at 'assertion failed: `(left == right)`
//@error-in-other-file: left: `2`
//@error-in-other-file:right: `3`: 1 + 1 definitely should be 3'
//@ignore-target-emscripten no processes

fn main() {
    assert_eq!(1 + 1, 3, "1 + 1 definitely should be 3");
}
