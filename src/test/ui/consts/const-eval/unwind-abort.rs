#![feature(unwind_attributes, const_panic)]

#[unwind(aborts)]
const fn foo() {
    panic!() //~ ERROR evaluation of constant value failed
}

const _: () = foo();
// Ensure that the CTFE engine handles calls to `#[unwind(aborts)]` gracefully

fn main() {
    let _ = foo();
}
