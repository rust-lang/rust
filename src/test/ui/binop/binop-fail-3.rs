// run-fail
// error-pattern:quux

fn foo() -> ! {
    panic!("quux");
}

fn main() {
    foo() == foo(); // these types wind up being defaulted to ()
}
