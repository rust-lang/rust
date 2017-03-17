// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A support library for macro authors when defining new macros.
//!
//! This library, provided by the standard distribution, provides the types
//! consumed in the interfaces of procedurally defined macro definitions.
//! Currently the primary use of this crate is to provide the ability to define
//! new custom derive modes through `#[proc_macro_derive]`.
//!
//! Note that this crate is intentionally very bare-bones currently. The main
//! type, `TokenStream`, only supports `fmt::Display` and `FromStr`
//! implementations, indicating that it can only go to and come from a string.
//! This functionality is intended to be expanded over time as more surface
//! area for macro authors is stabilized.
//!
//! See [the book](../book/first-edition/procedural-macros.html) for more.

#![crate_name = "proc_macro"]
#![stable(feature = "proc_macro_lib", since = "1.15.0")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![deny(warnings)]
#![deny(missing_docs)]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables, unused_mut))))]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(lang_items)]

extern crate syntax;

use std::fmt;
use std::str::FromStr;

use syntax::errors::DiagnosticBuilder;
use syntax::parse;
use syntax::tokenstream;

/// The main type provided by this crate, representing an abstract stream of
/// tokens.
///
/// This is both the input and output of `#[proc_macro_derive]` definitions.
/// Currently it's required to be a list of valid Rust items, but this
/// restriction may be lifted in the future.
///
/// The API of this type is intentionally bare-bones, but it'll be expanded over
/// time!
#[stable(feature = "proc_macro_lib", since = "1.15.0")]
pub struct TokenStream(tokenstream::TokenStream);

/// Error returned from `TokenStream::from_str`.
#[derive(Debug)]
#[stable(feature = "proc_macro_lib", since = "1.15.0")]
pub struct LexError {
    _inner: (),
}

/// Permanently unstable internal implementation details of this crate. This
/// should not be used.
///
/// These methods are used by the rest of the compiler to generate instances of
/// `TokenStream` to hand to macro definitions, as well as consume the output.
///
/// Note that this module is also intentionally separate from the rest of the
/// crate. This allows the `#[unstable]` directive below to naturally apply to
/// all of the contents.
#[unstable(feature = "proc_macro_internals", issue = "27812")]
#[doc(hidden)]
pub mod __internal {
    use std::cell::Cell;
    use std::rc::Rc;

    use syntax::ast;
    use syntax::ext::base::ExtCtxt;
    use syntax::ext::hygiene::Mark;
    use syntax::ptr::P;
    use syntax::parse::{self, token, ParseSess};
    use syntax::tokenstream;

    use super::{TokenStream, LexError};

    pub fn new_token_stream(item: P<ast::Item>) -> TokenStream {
        let (span, token) = (item.span, token::Interpolated(Rc::new(token::NtItem(item))));
        TokenStream(tokenstream::TokenTree::Token(span, token).into())
    }

    pub fn token_stream_wrap(inner: tokenstream::TokenStream) -> TokenStream {
        TokenStream(inner)
    }

    pub fn token_stream_parse_items(stream: TokenStream) -> Result<Vec<P<ast::Item>>, LexError> {
        with_sess(move |(sess, _)| {
            let mut parser = parse::stream_to_parser(sess, stream.0);
            let mut items = Vec::new();

            while let Some(item) = try!(parser.parse_item().map_err(super::parse_to_lex_err)) {
                items.push(item)
            }

            Ok(items)
        })
    }

    pub fn token_stream_inner(stream: TokenStream) -> tokenstream::TokenStream {
        stream.0
    }

    pub trait Registry {
        fn register_custom_derive(&mut self,
                                  trait_name: &str,
                                  expand: fn(TokenStream) -> TokenStream,
                                  attributes: &[&'static str]);

        fn register_attr_proc_macro(&mut self,
                                    name: &str,
                                    expand: fn(TokenStream, TokenStream) -> TokenStream);

        fn register_bang_proc_macro(&mut self,
                                    name: &str,
                                    expand: fn(TokenStream) -> TokenStream);
    }

    // Emulate scoped_thread_local!() here essentially
    thread_local! {
        static CURRENT_SESS: Cell<(*const ParseSess, Mark)> =
            Cell::new((0 as *const _, Mark::root()));
    }

    pub fn set_sess<F, R>(cx: &ExtCtxt, f: F) -> R
        where F: FnOnce() -> R
    {
        struct Reset { prev: (*const ParseSess, Mark) }

        impl Drop for Reset {
            fn drop(&mut self) {
                CURRENT_SESS.with(|p| p.set(self.prev));
            }
        }

        CURRENT_SESS.with(|p| {
            let _reset = Reset { prev: p.get() };
            p.set((cx.parse_sess, cx.current_expansion.mark));
            f()
        })
    }

    pub fn with_sess<F, R>(f: F) -> R
        where F: FnOnce((&ParseSess, Mark)) -> R
    {
        let p = CURRENT_SESS.with(|p| p.get());
        assert!(!p.0.is_null(), "proc_macro::__internal::with_sess() called \
                                 before set_parse_sess()!");
        f(unsafe { (&*p.0, p.1) })
    }
}

fn parse_to_lex_err(mut err: DiagnosticBuilder) -> LexError {
    err.cancel();
    LexError { _inner: () }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        __internal::with_sess(|(sess, mark)| {
            let src = src.to_string();
            let name = "<proc-macro source code>".to_string();
            let call_site = mark.expn_info().unwrap().call_site;
            let stream = parse::parse_stream_from_source_str(name, src, sess, Some(call_site));
            Ok(__internal::token_stream_wrap(stream))
        })
    }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
