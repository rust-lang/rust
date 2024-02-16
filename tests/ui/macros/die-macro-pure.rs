//@ run-fail
//@ error-pattern:test
//@ ignore-emscripten no processes

fn f() {
    panic!("test");
}

fn main() {
    f();
}
