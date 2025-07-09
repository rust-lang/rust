//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
