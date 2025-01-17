#![feature(proc_macro_tracked_env)]

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro::tracked_env::var;

#[proc_macro]
pub fn generate_const(input: TokenStream) -> TokenStream {
    let the_const = match var("THE_CONST") {
        Ok(x) if x == "12" => {
            "const THE_CONST: u32 = 12;"
        }
        _ => {
            "const THE_CONST: u32 = 0;"
        }
    };
    let another = if var("ANOTHER").is_ok() {
        "const ANOTHER: u32 = 1;"
    } else {
        "const ANOTHER: u32 = 2;"
    };
    format!("{the_const}{another}").parse().unwrap()
}
