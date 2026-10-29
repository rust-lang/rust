//@ run-fail
//@ regex-error-pattern: thread 'main' \(\d+\) panicked at
//@ needs-subprocess

fn main() {
    panic!()
}
