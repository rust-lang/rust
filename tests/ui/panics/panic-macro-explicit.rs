// run-fail
//@error-in-other-file:panicked at 'explicit panic'
//@ignore-target-emscripten no processes

fn main() {
    panic!();
}
