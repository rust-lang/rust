// run-fail
//@error-in-other-file:test
//@ignore-target-emscripten no processes

fn f() {
    panic!("test");
}

fn main() {
    f();
}
