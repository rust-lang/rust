// run-fail
// error-pattern:panicked at 'assertion failed: `(left != right)`
// error-pattern: left: `2`
// error-pattern:right: `2`: 1 + 1 definitely should not be 2'
// ignore-emscripten no processes

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
