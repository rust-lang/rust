//@ run-fail
//@ error-pattern:panic evaluated
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    // This used to trigger an LLVM assertion during compilation
    let x = [panic!("panic evaluated"); 2];
}
