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
//! new custom derive modes through `#[rustc_macro_derive]`.
//!
//! Added recently as part of [RFC 1681] this crate is currently *unstable* and
//! requires the `#![feature(rustc_macro_lib)]` directive to use. Eventually,
//! though, it is intended for this crate to become stable to use (perhaps under
//! a different name).
//!
//! [RFC 1681]: https://github.com/rust-lang/rfcs/blob/master/text/1681-macros-1.1.md
//!
//! Note that this crate is intentionally very bare-bones currently. The main
//! type, `TokenStream`, only supports `fmt::Display` and `FromStr`
//! implementations, indicating that it can only go to and come from a string.
//! This functionality is intended to be expanded over time as more surface
//! area for macro authors is stabilized.

#![crate_name = "rustc_macro"]
#![unstable(feature = "rustc_macro_lib", issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![cfg_attr(not(stage0), deny(warnings))]
#![deny(missing_docs)]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(lang_items)]

extern crate syntax;

use std::fmt;
use std::str::FromStr;

use syntax::ast;
use syntax::parse;
use syntax::ptr::P;

/// The main type provided by this crate, representing an abstract stream of
/// tokens.
///
/// This is both the input and output of `#[rustc_macro_derive]` definitions.
/// Currently it's required to be a list of valid Rust items, but this
/// restriction may be lifted in the future.
///
/// The API of this type is intentionally bare-bones, but it'll be expanded over
/// time!
pub struct TokenStream {
    inner: Vec<P<ast::Item>>,
}

/// Error returned from `TokenStream::from_str`.
#[derive(Debug)]
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
#[unstable(feature = "rustc_macro_internals", issue = "27812")]
#[doc(hidden)]
pub mod __internal {
    use std::cell::Cell;

    use syntax::ast;
    use syntax::ptr::P;
    use syntax::parse::ParseSess;
    use super::TokenStream;

    pub fn new_token_stream(item: P<ast::Item>) -> TokenStream {
        TokenStream { inner: vec![item] }
    }

    pub fn token_stream_items(stream: TokenStream) -> Vec<P<ast::Item>> {
        stream.inner
    }

    pub trait Registry {
        fn register_custom_derive(&mut self,
                                  trait_name: &str,
                                  expand: fn(TokenStream) -> TokenStream);
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
        assert!(!p.is_null());
        f(unsafe { &*p })
    }
}

impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        __internal::with_parse_sess(|sess| {
            let src = src.to_string();
            let cfg = Vec::new();
            let name = "rustc-macro source code".to_string();
            let mut parser = parse::new_parser_from_source_str(sess, cfg, name,
                                                               src);
            let mut ret = TokenStream { inner: Vec::new() };
            loop {
                match parser.parse_item() {
                    Ok(Some(item)) => ret.inner.push(item),
                    Ok(None) => return Ok(ret),
                    Err(mut err) => {
                        err.cancel();
                        return Err(LexError { _inner: () })
                    }
                }
            }
        })
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for item in self.inner.iter() {
            let item = syntax::print::pprust::item_to_string(item);
            try!(f.write_str(&item));
            try!(f.write_str("\n"));
        }
        Ok(())
    }
}
