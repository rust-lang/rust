#![feature(proc_macro_tracked, proc_macro_tracked_env_var)]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn access_env_vars(_: TokenStream) -> TokenStream {
    let _ = tracked::env::var("EXISTING_PROC_MACRO_ENV");
    let _ = tracked::env::var("NONEXISTENT_PROC_MACEO_ENV");
    TokenStream::new()
}
