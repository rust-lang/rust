//! Check that we correctly lint procedural macros.

extern crate proc_macro;

use proc_macro::TokenStream;

#[allow(dead_code)]
fn f() {
    let _x = 3.14;
    //~^ ERROR: approximate value of `f{32, 64}::consts::PI` found
}

#[proc_macro]
pub fn mybangmacro(t: TokenStream) -> TokenStream {
    t
}

#[proc_macro_derive(MyDerivedTrait)]
pub fn myderive(t: TokenStream) -> TokenStream {
    t
}

#[proc_macro_attribute]
pub fn myattribute(t: TokenStream, a: TokenStream) -> TokenStream {
    t
}
