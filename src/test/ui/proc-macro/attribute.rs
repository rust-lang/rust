// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive] //~ ERROR malformed `proc_macro_derive` attribute
pub fn foo1(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive = ""] //~ ERROR malformed `proc_macro_derive` attribute
pub fn foo2(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d3, a, b)]
//~^ ERROR attribute must have either one or two arguments
pub fn foo3(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d4, attributes(a), b)]
//~^ ERROR attribute must have either one or two arguments
pub fn foo4(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive("a")]
//~^ ERROR: not a meta item
pub fn foo5(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d6 = "")]
//~^ ERROR: must only be one word
pub fn foo6(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(m::d7)]
//~^ ERROR: must only be one word
pub fn foo7(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d8(a))]
//~^ ERROR: must only be one word
pub fn foo8(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(self)]
//~^ ERROR: `self` cannot be a name of derive macro
pub fn foo9(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(PartialEq)]
//~^ ERROR: cannot override a built-in derive macro
pub fn foo10(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d11, a)]
//~^ ERROR: second argument must be `attributes`
//~| ERROR: attribute must be of form: `attributes(foo, bar)`
pub fn foo11(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d12, attributes)]
//~^ ERROR: attribute must be of form: `attributes(foo, bar)`
pub fn foo12(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d13, attributes("a"))]
//~^ ERROR: not a meta item
pub fn foo13(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d14, attributes(a = ""))]
//~^ ERROR: must only be one word
pub fn foo14(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d15, attributes(m::a))]
//~^ ERROR: must only be one word
pub fn foo15(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d16, attributes(a(b)))]
//~^ ERROR: must only be one word
pub fn foo16(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d17, attributes(self))]
//~^ ERROR: `self` cannot be a name of derive helper attribute
pub fn foo17(input: TokenStream) -> TokenStream { input }
