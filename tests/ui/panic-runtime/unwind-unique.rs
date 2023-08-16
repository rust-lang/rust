// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
