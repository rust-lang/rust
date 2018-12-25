// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro = "test"] //~ ERROR: does not take any arguments
pub fn a(a: TokenStream) -> TokenStream { a }

#[proc_macro()] //~ ERROR: does not take any arguments
pub fn c(a: TokenStream) -> TokenStream { a }

#[proc_macro(x)] //~ ERROR: does not take any arguments
pub fn d(a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute = "test"] //~ ERROR: does not take any arguments
pub fn e(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute()] //~ ERROR: does not take any arguments
pub fn g(_: TokenStream, a: TokenStream) -> TokenStream { a }

#[proc_macro_attribute(x)] //~ ERROR: does not take any arguments
pub fn h(_: TokenStream, a: TokenStream) -> TokenStream { a }
