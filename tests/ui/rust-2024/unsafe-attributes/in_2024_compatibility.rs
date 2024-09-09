#![deny(rust_2024_compatibility)]

#[no_mangle]
//~^ ERROR: unsafe attribute used without unsafe
//~| WARN this is accepted in the current edition
extern "C" fn foo() {}

fn main() {}
