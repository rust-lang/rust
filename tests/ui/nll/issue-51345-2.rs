//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: explicit panic
//@ needs-subprocess

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
