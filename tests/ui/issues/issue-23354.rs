//@ run-fail
//@ error-pattern:panic evaluated
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
