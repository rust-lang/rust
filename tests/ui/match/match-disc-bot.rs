//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn f() -> ! {
    panic!("quux")
}
fn g() -> isize {
    match f() {
        true => 1,
        false => 0,
    }
}
fn main() {
    g();
}
