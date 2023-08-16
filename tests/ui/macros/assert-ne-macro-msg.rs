// run-fail
//@error-in-other-file:panicked at 'assertion failed: `(left != right)`
//@error-in-other-file: left: `2`
//@error-in-other-file:right: `2`: 1 + 1 definitely should not be 2'
//@ignore-target-emscripten no processes

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
