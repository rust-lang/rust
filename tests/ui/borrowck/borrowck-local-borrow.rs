//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let x = 2;
    let y = &x;
    panic!("panic 1");
}
