//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn a() {}

fn b() {
    panic!();
}

fn main() {
    let _x = vec![0];
    a();
    let _y = vec![0];
    b();
}
