//@ run-fail
//@ error-pattern:test
//@ needs-subprocess
// Just testing that panic!() type checks in statement or expr

fn f() {
    let __isize: isize = panic!("test");

    panic!();
}

fn main() {
    f();
}
