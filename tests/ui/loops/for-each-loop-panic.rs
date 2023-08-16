// run-fail
//@error-in-other-file:moop
//@ignore-target-emscripten no processes

fn main() {
    for _ in 0_usize..10_usize {
        panic!("moop");
    }
}
