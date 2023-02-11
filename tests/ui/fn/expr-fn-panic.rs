// run-fail
// error-pattern:explicit panic
// ignore-emscripten no processes

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
