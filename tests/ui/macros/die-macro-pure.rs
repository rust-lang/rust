//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn f() {
    panic!("test");
}

fn main() {
    f();
}
