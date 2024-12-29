//@ run-fail
//@ check-run-results:quux
//@ ignore-emscripten no processes

fn foo() -> ! {
    panic!("quux");
}

fn main() {
    foo() == foo(); // these types wind up being defaulted to ()
}
