//@ dont-require-annotations: NOTE

const extern "C" fn foo() {
    panic!() //~ NOTE inside `foo`
}

const _: () = foo(); //~ ERROR explicit panic
// Ensure that the CTFE engine handles calls to `extern "C"` aborting gracefully

fn main() {
    foo();
}
