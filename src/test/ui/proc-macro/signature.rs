// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

extern crate proc_macro;

#[proc_macro_derive(A)]
pub unsafe extern "C" fn foo(a: i32, b: u32) -> u32 {
    //~^ ERROR: expected a `Fn<(proc_macro::TokenStream,)>` closure, found `unsafe extern "C" fn
    loop {}
}
