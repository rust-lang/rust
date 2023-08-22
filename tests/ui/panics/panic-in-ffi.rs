// run-fail
// check-run-results
// error-pattern: panic in a function that cannot unwind
// normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
// normalize-stderr-test: "\n +at [^\n]+" -> ""
// ignore-emscripten no processes
#![feature(c_unwind)]

extern "C" fn panic_in_ffi() {
    panic!("Test");
}

fn main() {
    panic_in_ffi();
}
