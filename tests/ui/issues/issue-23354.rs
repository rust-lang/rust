//@ run-fail
//@ error-pattern:panic evaluated
//@ ignore-emscripten no processes

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
