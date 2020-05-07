// run-fail
// error-pattern:panicked at 'explicit panic'
// ignore-emscripten no processes

fn main() {
    panic!();
}
