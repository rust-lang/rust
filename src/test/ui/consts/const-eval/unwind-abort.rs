#![feature(unwind_attributes, const_panic)]

#[unwind(aborts)]
const fn foo() {
    panic!() //~ evaluation of constant value failed
}

const _: () = foo(); //~ any use of this value will cause an error
// Ensure that the CTFE engine handles calls to `#[unwind(aborts)]` gracefully

fn main() {
    let _ = foo();
}
