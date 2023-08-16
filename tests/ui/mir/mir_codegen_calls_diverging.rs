// run-fail
//@error-in-other-file:diverging_fn called
//@ignore-target-emscripten no processes

fn diverging_fn() -> ! {
    panic!("diverging_fn called")
}

fn mir() {
    diverging_fn();
}

fn main() {
    mir();
}
