//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
