#![feature(unwind_attributes, const_panic)]

#[unwind(aborts)]
const fn foo() {
    panic!() //~ ERROR any use of this value will cause an error [const_err]
    //~| WARN this was previously accepted by the compiler but is being phased out
}

const _: () = foo();
// Ensure that the CTFE engine handles calls to `#[unwind(aborts)]` gracefully

fn main() {
    let _ = foo();
}
