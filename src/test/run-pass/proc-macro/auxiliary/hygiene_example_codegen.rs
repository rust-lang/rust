// force-host
// no-prefer-dynamic

#![feature(proc_macro_quote, proc_macro_hygiene)]
#![crate_type = "proc-macro"]

extern crate proc_macro as proc_macro_renamed; // This does not break `quote!`

use proc_macro_renamed::{TokenStream, quote};

#[proc_macro]
pub fn hello(input: TokenStream) -> TokenStream {
    quote!(hello_helper!($input))
    //^ `hello_helper!` always resolves to the following proc macro,
    //| no matter where `hello!` is used.
}

#[proc_macro]
pub fn hello_helper(input: TokenStream) -> TokenStream {
    quote! {
        extern crate hygiene_example; // This is never a conflict error
        let string = format!("hello {}", $input);
        //^ `format!` always resolves to the prelude macro,
        //| even if a different `format!` is in scope where `hello!` is used.
        hygiene_example::print(&string)
    }
}
