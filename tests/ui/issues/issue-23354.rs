//@ run-fail
//@ check-run-results
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
