// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
