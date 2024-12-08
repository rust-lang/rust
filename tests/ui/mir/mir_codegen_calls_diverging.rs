//@ run-fail
//@ error-pattern:diverging_fn called
//@ ignore-emscripten no processes

fn diverging_fn() -> ! {
    panic!("diverging_fn called")
}

fn mir() {
    diverging_fn();
}

fn main() {
    mir();
}
