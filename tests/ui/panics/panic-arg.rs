//@ run-fail
//@ error-pattern:woe
//@ needs-subprocess

fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
