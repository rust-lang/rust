// run-fail
//@error-in-other-file:explicit panic
//@ignore-target-emscripten no processes

fn a() {}

fn b() {
    panic!();
}

fn main() {
    let _x = vec![0];
    a();
    let _y = vec![0];
    b();
}
