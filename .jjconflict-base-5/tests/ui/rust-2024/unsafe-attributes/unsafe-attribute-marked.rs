//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@ check-pass


#[unsafe(no_mangle)]
extern "C" fn foo() {}

fn main() {}
