//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2021] check-pass
//@[edition2024] edition:2024


#[no_mangle] //[edition2024]~ ERROR: unsafe attribute used without unsafe
extern "C" fn foo() {}

fn main() {}
