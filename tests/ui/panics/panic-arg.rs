//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
