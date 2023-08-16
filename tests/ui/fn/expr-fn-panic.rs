// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
