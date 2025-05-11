//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

pub fn main() {
    panic!();
    println!("{}", 1);
}
