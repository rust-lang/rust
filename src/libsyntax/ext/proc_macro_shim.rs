// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is a shim file to ease the transition to the final procedural macro interface for
//! Macros 2.0. It currently exposes the `libsyntax` operations that the quasiquoter's
//! output needs to compile correctly, along with the following operators:
//!
//! - `build_block_emitter`, which produces a `block` output macro result from the
//!   provided TokenStream.

use ast;
use codemap::Span;
use parse::parser::Parser;
use ptr::P;
use tokenstream::TokenStream;
use ext::base::*;

/// Take a `ExtCtxt`, `Span`, and `TokenStream`, and produce a Macro Result that parses
/// the TokenStream as a block and returns it as an `Expr`.
pub fn build_block_emitter<'cx>(cx: &'cx mut ExtCtxt,
                                sp: Span,
                                output: TokenStream)
                                -> Box<MacResult + 'cx> {
    let parser = cx.new_parser_from_tts(&output.to_tts());

    struct Result<'a> {
        prsr: Parser<'a>,
        span: Span,
    }; //FIXME is this the right lifetime

    impl<'a> Result<'a> {
        fn block(&mut self) -> P<ast::Block> {
            let res = self.prsr.parse_block().unwrap();
            res
        }
    }

    impl<'a> MacResult for Result<'a> {
        fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
            let mut me = *self;
            Some(P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprKind::Block(me.block()),
                span: me.span,
                attrs: ast::ThinVec::new(),
            }))

        }
    }

    Box::new(Result {
        prsr: parser,
        span: sp,
    })
}

pub mod prelude {
    pub use super::build_block_emitter;
    pub use ast::Ident;
    pub use codemap::{DUMMY_SP, Span};
    pub use ext::base::{ExtCtxt, MacResult};
    pub use parse::token::{self, Token, DelimToken, keywords, str_to_ident};
    pub use tokenstream::{TokenTree, TokenStream};
}
