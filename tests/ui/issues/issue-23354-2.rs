// run-fail
//@error-in-other-file:panic evaluated
//@ignore-target-emscripten no processes

#[allow(unused_variables)]
fn main() {
    // This used to trigger an LLVM assertion during compilation
    let x = [panic!("panic evaluated"); 2];
}
