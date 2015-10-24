// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr;
use ast;
use codemap::{spanned, Spanned, mk_sp, Span};
use parse::common::*; //resolve bug?
use parse::PResult;
use parse::token;
use parse::parser::{Parser, TokenType};
use ptr::P;

impl<'a> Parser<'a> {
    /// Parse attributes that appear before an item
    pub fn parse_outer_attributes(&mut self) -> PResult<Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        loop {
            debug!("parse_outer_attributes: self.token={:?}",
                   self.token);
            match self.token {
              token::Pound => {
                attrs.push(try!(self.parse_attribute(false)));
              }
              token::DocComment(s) => {
                let attr = ::attr::mk_sugared_doc_attr(
                    attr::mk_attr_id(),
                    self.id_to_interned_str(ast::Ident::with_empty_ctxt(s)),
                    self.span.lo,
                    self.span.hi
                );
                if attr.node.style != ast::AttrStyle::Outer {
                  return Err(self.fatal("expected outer comment"));
                }
                attrs.push(attr);
                try!(self.bump());
              }
              _ => break
            }
        }
        return Ok(attrs);
    }

    /// Matches `attribute = # ! [ meta_item ]`
    ///
    /// If permit_inner is true, then a leading `!` indicates an inner
    /// attribute
    fn parse_attribute(&mut self, permit_inner: bool) -> PResult<ast::Attribute> {
        debug!("parse_attributes: permit_inner={:?} self.token={:?}",
               permit_inner, self.token);
        let (span, value, mut style) = match self.token {
            token::Pound => {
                let lo = self.span.lo;
                try!(self.bump());

                if permit_inner { self.expected_tokens.push(TokenType::Token(token::Not)); }
                let style = if self.token == token::Not {
                    try!(self.bump());
                    if !permit_inner {
                        let span = self.span;
                        self.span_err(span,
                                      "an inner attribute is not permitted in \
                                       this context");
                        self.fileline_help(span,
                                       "place inner attribute at the top of the module or block");
                    }
                    ast::AttrStyle::Inner
                } else {
                    ast::AttrStyle::Outer
                };

                try!(self.expect(&token::OpenDelim(token::Bracket)));
                let meta_item = try!(self.parse_meta_item());
                let hi = self.span.hi;
                try!(self.expect(&token::CloseDelim(token::Bracket)));

                (mk_sp(lo, hi), meta_item, style)
            }
            _ => {
                let token_str = self.this_token_to_string();
                return Err(self.fatal(&format!("expected `#`, found `{}`", token_str)));
            }
        };

        if permit_inner && self.token == token::Semi {
            try!(self.bump());
            self.span_warn(span, "this inner attribute syntax is deprecated. \
                           The new syntax is `#![foo]`, with a bang and no semicolon");
            style = ast::AttrStyle::Inner;
        }

        Ok(Spanned {
            span: span,
            node: ast::Attribute_ {
                id: attr::mk_attr_id(),
                style: style,
                value: value,
                is_sugared_doc: false
            }
        })
    }

    /// Parse attributes that appear after the opening of an item. These should
    /// be preceded by an exclamation mark, but we accept and warn about one
    /// terminated by a semicolon.

    /// matches inner_attrs*
    pub fn parse_inner_attributes(&mut self) -> PResult<Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = vec![];
        loop {
            match self.token {
                token::Pound => {
                    // Don't even try to parse if it's not an inner attribute.
                    if !self.look_ahead(1, |t| t == &token::Not) {
                        break;
                    }

                    let attr = try!(self.parse_attribute(true));
                    assert!(attr.node.style == ast::AttrStyle::Inner);
                    attrs.push(attr);
                }
                token::DocComment(s) => {
                    // we need to get the position of this token before we bump.
                    let Span { lo, hi, .. } = self.span;
                    let str = self.id_to_interned_str(ast::Ident::with_empty_ctxt(s));
                    let attr = attr::mk_sugared_doc_attr(attr::mk_attr_id(), str, lo, hi);
                    if attr.node.style == ast::AttrStyle::Inner {
                        attrs.push(attr);
                        try!(self.bump());
                    } else {
                        break;
                    }
                }
                _ => break
            }
        }
        Ok(attrs)
    }

    /// matches meta_item = IDENT
    /// | IDENT = lit
    /// | IDENT meta_seq
    pub fn parse_meta_item(&mut self) -> PResult<P<ast::MetaItem>> {
        let nt_meta = match self.token {
            token::Interpolated(token::NtMeta(ref e)) => {
                Some(e.clone())
            }
            _ => None
        };

        match nt_meta {
            Some(meta) => {
                try!(self.bump());
                return Ok(meta);
            }
            None => {}
        }

        let lo = self.span.lo;
        let ident = try!(self.parse_ident());
        let name = self.id_to_interned_str(ident);
        match self.token {
            token::Eq => {
                try!(self.bump());
                let lit = try!(self.parse_lit());
                // FIXME #623 Non-string meta items are not serialized correctly;
                // just forbid them for now
                match lit.node {
                    ast::LitStr(..) => {}
                    _ => {
                        self.span_err(
                            lit.span,
                            "non-string literals are not allowed in meta-items");
                    }
                }
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaNameValue(name, lit))))
            }
            token::OpenDelim(token::Paren) => {
                let inner_items = try!(self.parse_meta_seq());
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaList(name, inner_items))))
            }
            _ => {
                let hi = self.last_span.hi;
                Ok(P(spanned(lo, hi, ast::MetaWord(name))))
            }
        }
    }

    /// matches meta_seq = ( COMMASEP(meta_item) )
    fn parse_meta_seq(&mut self) -> PResult<Vec<P<ast::MetaItem>>> {
        self.parse_unspanned_seq(&token::OpenDelim(token::Paren),
                                 &token::CloseDelim(token::Paren),
                                 seq_sep_trailing_allowed(token::Comma),
                                 |p| p.parse_meta_item())
    }
}
