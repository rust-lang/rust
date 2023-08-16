// run-fail
//@error-in-other-file:panic 1
//@ignore-target-emscripten no processes

fn main() {
    let x = 2;
    let y = &x;
    panic!("panic 1");
}
