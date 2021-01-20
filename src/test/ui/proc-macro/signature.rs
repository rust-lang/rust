// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

extern crate proc_macro;

#[proc_macro_derive(A)]
pub unsafe extern fn foo(a: i32, b: u32) -> u32 {
    //~^ ERROR: mismatched types
    loop {}
}
