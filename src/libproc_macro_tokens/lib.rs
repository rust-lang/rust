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
//! This crate provides the prelude (at libproc_macro_tokens::prelude), which
//! provides a number of operations:
//! - `concat`, for concatenating two TokenStreams.
//! - `ident_eq`, for checking if two identifiers are equal regardless of syntax context.
//! - `str_to_token_ident`, for converting an `&str` into a Token.
//! - `keyword_to_token_delim`, for converting a `parse::token::keywords::Keyword` into a
//!    Token.
//! - `build_delimited`, for creating a new TokenStream from an existing one and a delimiter
//!    by wrapping the TokenStream in the delimiter.
//! - `build_bracket_delimited`, `build_brace_delimited`, and `build_paren_delimited`, for
//!    easing the above.
//! - `build_empty_args`, which returns a TokenStream containing `()`.
//! - `lex`, which takes an `&str` and returns the TokenStream it represents.
//!
//! ## TokenStreams
//!
//! TokenStreams serve as the basis of the macro system. They are, in essence, vectors of
//! TokenTrees, where indexing treats delimited values as a single term. That is, the term
//! `even(a+c) && even(b)` will be indexibly encoded as `even | (a+c) | even | (b)` where,
//! in reality, `(a+c)` is actually a decorated pointer to `a | + | c`.
//!
//! If a user has a TokenStream that is a single, delimited value, they can use
//! `maybe_delimited` to destruct it and receive the internal vector as a new TokenStream
//! as:
//! ```
//! `(a+c)`.maybe_delimited() ~> Some(a | + | c)`
//! ```
//!
//! Check the TokenStream documentation for more information; the structure also provides
//! cheap concatenation and slicing.
//!

#![crate_name = "proc_macro_tokens"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(staged_api)]
#![feature(rustc_private)]

extern crate syntax;

pub mod build;
pub mod parse;
pub mod prelude;
