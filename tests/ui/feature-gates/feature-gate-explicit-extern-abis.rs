#![feature(explicit_extern_abis)]

extern fn foo() {}
//~^ ERROR extern declarations without an explicit ABI are disallowed

fn main() {}
