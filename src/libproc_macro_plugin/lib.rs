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
//! This crate provides the `qquote!` macro for syntax creation.
//!
//! The `qquote!` macro imports `syntax::ext::proc_macro_shim::prelude::*`, so you
//! will need to `extern crate syntax` for usage. (This is a temporary solution until more
//! of the external API in libproc_macro_tokens is stabilized to support the token construction
//! operations that the qausiquoter relies on.) The shim file also provides additional
//! operations, such as `build_block_emitter` (as used in the `cond` example below).
//!
//! ## Quasiquotation
//!
//! The quasiquoter creates output that, when run, constructs the tokenstream specified as
//! input. For example, `qquote!(5 + 5)` will produce a program, that, when run, will
//! construct the TokenStream `5 | + | 5`.
//!
//! ### Unquoting
//!
//! Unquoting is currently done as `unquote`, and works by taking the single next
//! TokenTree in the TokenStream as the unquoted term. Ergonomically, `unquote(foo)` works
//! fine, but `unquote foo` is also supported.
//!
//! A simple example might be:
//!
//!```
//!fn double(tmp: TokenStream) -> TokenStream {
//!    qquote!(unquote(tmp) * 2)
//!}
//!```
//!
//! ### Large Example: Implementing Scheme's `cond`
//!
//! Below is the full implementation of Scheme's `cond` operator.
//!
//! ```
//! fn cond_rec(input: TokenStream) -> TokenStream {
//!   if input.is_empty() { return quote!(); }
//!
//!   let next = input.slice(0..1);
//!   let rest = input.slice_from(1..);
//!
//!   let clause : TokenStream = match next.maybe_delimited() {
//!     Some(ts) => ts,
//!     _ => panic!("Invalid input"),
//!   };
//!
//!   // clause is ([test]) [rhs]
//!   if clause.len() < 2 { panic!("Invalid macro usage in cond: {:?}", clause) }
//!
//!   let test: TokenStream = clause.slice(0..1);
//!   let rhs: TokenStream = clause.slice_from(1..);
//!
//!   if ident_eq(&test[0], str_to_ident("else")) || rest.is_empty() {
//!     quote!({unquote(rhs)})
//!   } else {
//!     quote!({if unquote(test) { unquote(rhs) } else { cond!(unquote(rest)) } })
//!   }
//! }
//! ```
//!

#![crate_name = "proc_macro_plugin"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![feature(plugin_registrar)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(staged_api)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]

extern crate rustc_plugin;
extern crate syntax;
extern crate proc_macro_tokens;
#[macro_use] extern crate log;

mod qquote;

use qquote::qquote;

use rustc_plugin::Registry;

// ____________________________________________________________________________________________
// Main macro definition

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("qquote", qquote);
}
