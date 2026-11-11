//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: --crate-type proc-macro

#![crate_type="proc-macro"]
#![crate_name="some_macros"]

extern crate proc_macro;

use proc_macro::TokenStream;

macro_rules! make_attr_macro {
    ($name:ident) => {
        /// Generated doc comment
        #[proc_macro_attribute]
        pub fn $name(args: TokenStream, input: TokenStream) -> TokenStream {
            panic!()
        }
    }
}

make_attr_macro!(first_attr);
make_attr_macro!(second_attr);

/// a proc-macro that swallows its input and does nothing.
#[proc_macro]
pub fn some_proc_macro(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

/// a proc-macro attribute that passes its item through verbatim.
#[proc_macro_attribute]
pub fn some_proc_attr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// a derive attribute that adds nothing to its input.
#[proc_macro_derive(SomeDerive)]
pub fn some_derive(_item: TokenStream) -> TokenStream {
    TokenStream::new()
}

/// Doc comment from the original crate
#[proc_macro]
pub fn reexported_macro(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
