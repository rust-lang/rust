// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

#[proc_macro_derive]
//~^ ERROR: attribute must be of form: #[proc_macro_derive(TraitName)]
pub fn foo1(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive = "foo"]
//~^ ERROR: attribute must be of form: #[proc_macro_derive(TraitName)]
pub fn foo2(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(
    a = "b"
)]
//~^^ ERROR: must only be one word
pub fn foo3(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(b, c, d)]
//~^ ERROR: attribute must have either one or two arguments
pub fn foo4(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(d(e))]
//~^ ERROR: must only be one word
pub fn foo5(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(f, attributes(g = "h"))]
//~^ ERROR: must only be one word
pub fn foo6(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(i, attributes(j(k)))]
//~^ ERROR: must only be one word
pub fn foo7(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}

#[proc_macro_derive(l, attributes(m), n)]
//~^ ERROR: attribute must have either one or two arguments
pub fn foo8(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    input
}
