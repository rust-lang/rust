//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

fn f() -> ! {
    panic!()
}

fn main() {
    f();
}
