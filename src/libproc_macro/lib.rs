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
//! See [the book](../../book/procedural-macros.html) for more.

#![crate_name = "proc_macro"]
#![stable(feature = "proc_macro_lib", since = "1.15.0")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![deny(warnings)]
#![deny(missing_docs)]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(lang_items)]

extern crate syntax;

use std::fmt;
use std::str::FromStr;

use syntax::errors::DiagnosticBuilder;
use syntax::parse;
use syntax::tokenstream::TokenStream as TokenStream_;

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
pub struct TokenStream {
    inner: TokenStream_,
}

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
    use syntax::ptr::P;
    use syntax::parse::{self, token, ParseSess};
    use syntax::tokenstream::TokenStream as TokenStream_;

    use super::{TokenStream, LexError};

    pub fn new_token_stream(item: P<ast::Item>) -> TokenStream {
        TokenStream { inner: TokenStream_::from_tokens(vec![
            token::Interpolated(Rc::new(token::NtItem(item)))
        ])}
    }

    pub fn token_stream_wrap(inner: TokenStream_) -> TokenStream {
        TokenStream {
            inner: inner
        }
    }

    pub fn token_stream_parse_items(stream: TokenStream) -> Result<Vec<P<ast::Item>>, LexError> {
        with_parse_sess(move |sess| {
            let mut parser = parse::new_parser_from_ts(sess, stream.inner);
            let mut items = Vec::new();

            while let Some(item) = try!(parser.parse_item().map_err(super::parse_to_lex_err)) {
                items.push(item)
            }

            Ok(items)
        })
    }

    pub fn token_stream_inner(stream: TokenStream) -> TokenStream_ {
        stream.inner
    }

    pub trait Registry {
        fn register_custom_derive(&mut self,
                                  trait_name: &str,
                                  expand: fn(TokenStream) -> TokenStream,
                                  attributes: &[&'static str]);

        fn register_attr_proc_macro(&mut self,
                                    name: &str,
                                    expand: fn(TokenStream, TokenStream) -> TokenStream);
    }

    // Emulate scoped_thread_local!() here essentially
    thread_local! {
        static CURRENT_SESS: Cell<*const ParseSess> = Cell::new(0 as *const _);
    }

    pub fn set_parse_sess<F, R>(sess: &ParseSess, f: F) -> R
        where F: FnOnce() -> R
    {
        struct Reset { prev: *const ParseSess }

        impl Drop for Reset {
            fn drop(&mut self) {
                CURRENT_SESS.with(|p| p.set(self.prev));
            }
        }

        CURRENT_SESS.with(|p| {
            let _reset = Reset { prev: p.get() };
            p.set(sess);
            f()
        })
    }

    pub fn with_parse_sess<F, R>(f: F) -> R
        where F: FnOnce(&ParseSess) -> R
    {
        let p = CURRENT_SESS.with(|p| p.get());
        assert!(!p.is_null(), "proc_macro::__internal::with_parse_sess() called \
                               before set_parse_sess()!");
        f(unsafe { &*p })
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
        __internal::with_parse_sess(|sess| {
            let src = src.to_string();
            let name = "<proc-macro source code>".to_string();
            let tts = try!(parse::parse_tts_from_source_str(name, src, sess)
                .map_err(parse_to_lex_err));

            Ok(__internal::token_stream_wrap(TokenStream_::from_tts(tts)))
        })
    }
}

#[stable(feature = "proc_macro_lib", since = "1.15.0")]
impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}
