//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn main() {
    panic!();
}
