//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro = "test"]
//~^ ERROR malformed `proc_macro` attribute
//~| NOTE didn't expect any arguments here
pub fn a(a: TokenStream) -> TokenStream { a }

#[proc_macro()]
//~^ ERROR malformed `proc_macro` attribute
//~| NOTE didn't expect any arguments here
pub fn c(a: TokenStream) -> TokenStream { a }

#[proc_macro(x)]
//~^ ERROR malformed `proc_macro` attribute
//~| NOTE didn't expect any arguments here
pub fn d(a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute = "test"]
//~^ ERROR malformed `proc_macro_attribute` attribute
//~| NOTE didn't expect any arguments here
pub fn e(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute()]
//~^ ERROR malformed `proc_macro_attribute` attribute
//~| NOTE didn't expect any arguments here
pub fn g(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute(x)]
//~^ ERROR malformed `proc_macro_attribute` attribute
//~| NOTE didn't expect any arguments here
pub fn h(_: TokenStream, a: TokenStream) -> TokenStream { a }
