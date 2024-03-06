//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub extern "C" fn abi(a: TokenStream) -> TokenStream {
    //~^ ERROR function-like proc macro has incorrect signature
    a
}

#[proc_macro]
pub extern "system" fn abi2(a: TokenStream) -> TokenStream {
    //~^ ERROR function-like proc macro has incorrect signature
    a
}

#[proc_macro]
pub extern fn abi3(a: TokenStream) -> TokenStream {
    //~^ ERROR function-like proc macro has incorrect signature
    a
}

#[proc_macro]
pub extern "Rust" fn abi4(a: TokenStream) -> TokenStream {
    a
}
