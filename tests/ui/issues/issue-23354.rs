// run-fail
//@error-in-other-file:panic evaluated
//@ignore-target-emscripten no processes

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
