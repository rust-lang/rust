//@ run-fail
//@ error-pattern:thread 'main' panicked at
//@ needs-subprocess

fn main() {
    panic!()
}
