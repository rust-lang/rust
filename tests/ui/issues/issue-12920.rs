//@ run-fail
//@ check-run-results
//@ needs-subprocess

pub fn main() {
    panic!();
    println!("{}", 1);
}
