// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A generic trait to abstract the rewriting of an element (of the AST).

use syntax::codemap::{CodeMap, Span};
use syntax::parse::ParseSess;

use Shape;
use config::{Config, IndentStyle};

pub trait Rewrite {
    /// Rewrite self into shape.
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String>;
}

#[derive(Clone)]
pub struct RewriteContext<'a> {
    pub parse_session: &'a ParseSess,
    pub codemap: &'a CodeMap,
    pub config: &'a Config,
    pub inside_macro: bool,
    // Force block indent style even if we are using visual indent style.
    pub use_block: bool,
    // When `format_if_else_cond_comment` is true, unindent the comment on top
    // of the `else` or `else if`.
    pub is_if_else_block: bool,
    // When rewriting chain, veto going multi line except the last element
    pub force_one_line_chain: bool,
}

impl<'a> RewriteContext<'a> {
    pub fn snippet(&self, span: Span) -> String {
        self.codemap.span_to_snippet(span).unwrap()
    }

    /// Return true if we should use block indent style for rewriting function call.
    pub fn use_block_indent(&self) -> bool {
        self.config.fn_call_style() == IndentStyle::Block || self.use_block
    }
}
