// run-fail
//@error-in-other-file:quux
//@ignore-target-emscripten no processes

fn foo() -> ! {
    panic!("quux");
}

fn main() {
    foo() == foo(); // these types wind up being defaulted to ()
}
