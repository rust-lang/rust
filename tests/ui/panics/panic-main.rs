// run-fail
//@error-in-other-file:moop
//@ignore-target-emscripten no processes

fn main() {
    panic!("moop");
}
