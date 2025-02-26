//@ run-fail
//@ error-pattern:test
//@ needs-subprocess

fn f() {
    panic!("test");
}

fn main() {
    f();
}
