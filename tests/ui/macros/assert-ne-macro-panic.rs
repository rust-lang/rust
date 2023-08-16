// run-fail
//@error-in-other-file:assertion failed: `(left != right)`
//@error-in-other-file: left: `14`
//@error-in-other-file:right: `14`
//@ignore-target-emscripten no processes

fn main() {
    assert_ne!(14, 14);
}
