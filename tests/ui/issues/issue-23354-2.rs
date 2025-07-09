//@ run-fail
//@ check-run-results
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    // This used to trigger an LLVM assertion during compilation
    let x = [panic!("panic evaluated"); 2];
}
