//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected this to be a list
//~| NOTE for more information, visit
pub fn foo1(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive = ""]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected this to be a list
//~| NOTE for more information, visit
pub fn foo2(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d3, a, b)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE the only valid argument here is `attributes`
//~| NOTE for more information, visit
pub fn foo3(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d4, attributes(a), b)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
pub fn foo4(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive("a")]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect a literal here
//~| NOTE for more information, visit
pub fn foo5(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d6 = "")]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
pub fn foo6(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(m::d7)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
pub fn foo7(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d8(a))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
pub fn foo8(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(self)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
pub fn foo9(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(PartialEq)] // OK
pub fn foo10(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d11, a)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE the only valid argument here is `attributes`
//~| NOTE for more information, visit
pub fn foo11(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d12, attributes)]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected this to be a list
//~| NOTE for more information, visit
pub fn foo12(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d13, attributes("a"))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
pub fn foo13(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d14, attributes(a = ""))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
pub fn foo14(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d15, attributes(m::a))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
pub fn foo15(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d16, attributes(a(b)))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
pub fn foo16(input: TokenStream) -> TokenStream { input }

#[proc_macro_derive(d17, attributes(self))]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
pub fn foo17(input: TokenStream) -> TokenStream { input }
