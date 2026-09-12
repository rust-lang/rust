#![feature(proc_macro_tracked_env)]

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro::tracked::env_var;

#[proc_macro]
pub fn generate_const(input: TokenStream) -> TokenStream {
    let the_const = match env_var("THE_CONST") {
        Ok(x) => {
            format!("const THE_CONST: u32 = {x};")
        }
        _ => "const THE_CONST: u32 = 0;".to_string(),
    };
    let another = if env_var("ANOTHER").is_ok() {
        "const ANOTHER: u32 = 1;"
    } else {
        "const ANOTHER: u32 = 2;"
    };
    format!("{the_const}{another}").parse().unwrap()
}
