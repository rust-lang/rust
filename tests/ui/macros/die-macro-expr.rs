//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let __isize: isize = panic!("test");
}
