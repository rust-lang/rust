const extern "C" fn foo() {
    panic!() //~ inside `foo`
}

const _: () = foo(); //~ ERROR evaluation of constant value failed
// Ensure that the CTFE engine handles calls to `extern "C"` aborting gracefully

fn main() {
    foo();
}
