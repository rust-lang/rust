//@ run-fail
//@ error-pattern:test
//@ needs-subprocess

fn main() {
    let __isize: isize = panic!("test");
}
