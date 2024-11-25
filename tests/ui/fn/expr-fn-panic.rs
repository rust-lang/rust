//@ run-fail
//@ check-run-results:explicit panic
//@ ignore-emscripten no processes

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
