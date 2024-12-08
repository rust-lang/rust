extern crate proc_macro;

#[proc_macro_derive(Foo)]
//~^ ERROR: only usable with crates of the `proc-macro` crate type
pub fn foo(a: proc_macro::TokenStream) -> proc_macro::TokenStream {
    a
}

// Issue #37590
#[proc_macro_derive(Foo)]
//~^ ERROR: the `#[proc_macro_derive]` attribute may only be used on bare functions
pub struct Foo {
}

fn main() {}
