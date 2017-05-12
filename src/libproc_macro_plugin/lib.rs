// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Proc_Macro
//!
//! A library for procedural macro writers.
//!
//! ## Usage
//! This crate provides the `quote!` macro for syntax creation.
//!
//! The `quote!` macro uses the crate `syntax`, so users must declare `extern crate syntax;`
//! at the crate root. This is a temporary solution until we have better hygiene.
//!
//! ## Quasiquotation
//!
//! The quasiquoter creates output that, when run, constructs the tokenstream specified as
//! input. For example, `quote!(5 + 5)` will produce a program, that, when run, will
//! construct the TokenStream `5 | + | 5`.
//!
//! ### Unquoting
//!
//! Unquoting is done with `$`, and works by taking the single next ident as the unquoted term.
//! To quote `$` itself, use `$$`.
//!
//! A simple example is:
//!
//!```
//!fn double(tmp: TokenStream) -> TokenStream {
//!    quote!($tmp * 2)
//!}
//!```
//!
//! ### Large example: Scheme's `cond`
//!
//! Below is an example implementation of Scheme's `cond`.
//!
//! ```
//! fn cond(input: TokenStream) -> TokenStream {
//!     let mut conds = Vec::new();
//!     let mut input = input.trees().peekable();
//!     while let Some(tree) = input.next() {
//!         let mut cond = match tree {
//!             TokenTree::Delimited(_, ref delimited) => delimited.stream(),
//!             _ => panic!("Invalid input"),
//!         };
//!         let mut trees = cond.trees();
//!         let test = trees.next();
//!         let rhs = trees.collect::<TokenStream>();
//!         if rhs.is_empty() {
//!             panic!("Invalid macro usage in cond: {}", cond);
//!         }
//!         let is_else = match test {
//!             Some(TokenTree::Token(_, Token::Ident(ident))) if ident.name == "else" => true,
//!             _ => false,
//!         };
//!         conds.push(if is_else || input.peek().is_none() {
//!             quote!({ $rhs })
//!         } else {
//!             let test = test.unwrap();
//!             quote!(if $test { $rhs } else)
//!         });
//!     }
//!
//!     conds.into_iter().collect()
//! }
//! ```
#![crate_name = "proc_macro_plugin"]
#![cfg_attr(stage0, unstable(feature = "rustc_private", issue = "27812"))]
#![feature(plugin_registrar)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![cfg_attr(stage0, feature(staged_api))]
#![feature(rustc_diagnostic_macros)]
#![cfg_attr(stage0, feature(rustc_private))]

extern crate rustc_plugin;
extern crate syntax;
extern crate syntax_pos;

mod quote;
use quote::quote;

use rustc_plugin::Registry;
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;

// ____________________________________________________________________________________________
// Main macro definition

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_syntax_extension(Symbol::intern("quote"),
                                  SyntaxExtension::ProcMacro(Box::new(quote)));
}
