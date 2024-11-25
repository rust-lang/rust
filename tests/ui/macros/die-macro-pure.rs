//@ run-fail
//@ check-run-results:test
//@ ignore-emscripten no processes

fn f() {
    panic!("test");
}

fn main() {
    f();
}
