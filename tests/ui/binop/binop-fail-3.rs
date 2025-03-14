//@ run-fail
//@ error-pattern:quux
//@ needs-subprocess

fn foo() -> ! {
    panic!("quux");
}

fn main() {
    foo() == foo(); // these types wind up being defaulted to ()
}
