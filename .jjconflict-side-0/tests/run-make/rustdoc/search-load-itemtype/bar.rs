#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

//@ has bar/macro.a_procmacro.html
//@ hasraw search.index/name/*.js a_procmacro
#[proc_macro]
pub fn a_procmacro(_: TokenStream) -> TokenStream {
    unimplemented!()
}

//@ has bar/attr.a_procattribute.html
//@ hasraw search.index/name/*.js a_procattribute
#[proc_macro_attribute]
pub fn a_procattribute(_: TokenStream, _: TokenStream) -> TokenStream {
    unimplemented!()
}

//@ has bar/derive.AProcDerive.html
//@ !has bar/derive.a_procderive.html
//@ hasraw search.index/name/*.js AProcDerive
//@ !hasraw search.index/name/*.js a_procderive
#[proc_macro_derive(AProcDerive)]
pub fn a_procderive(_: TokenStream) -> TokenStream {
    unimplemented!()
}
