// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro = "test"] //~ ERROR malformed `proc_macro` attribute
pub fn a(a: TokenStream) -> TokenStream { a }

#[proc_macro()] //~ ERROR malformed `proc_macro` attribute
pub fn c(a: TokenStream) -> TokenStream { a }

#[proc_macro(x)] //~ ERROR malformed `proc_macro` attribute
pub fn d(a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute = "test"] //~ ERROR malformed `proc_macro_attribute` attribute
pub fn e(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute()] //~ ERROR malformed `proc_macro_attribute` attribute
pub fn g(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute(x)] //~ ERROR malformed `proc_macro_attribute` attribute
pub fn h(_: TokenStream, a: TokenStream) -> TokenStream { a }
