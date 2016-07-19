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
use syntax_pos::{mk_sp, Span};
use codemap::{spanned, Spanned};
use parse::common::SeqSep;
use parse::PResult;
use parse::token;
use parse::parser::{Parser, TokenType};
use ptr::P;

#[derive(PartialEq, Eq, Debug)]
enum InnerAttributeParsePolicy<'a> {
    Permitted,
    NotPermitted { reason: &'a str },
}

const DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG: &'static str = "an inner attribute is not \
                                                             permitted in this context";

impl<'a> Parser<'a> {
    /// Parse attributes that appear before an item
    pub fn parse_outer_attributes(&mut self) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        let mut just_parsed_doc_comment = false;
        loop {
            debug!("parse_outer_attributes: self.token={:?}", self.token);
            match self.token {
                token::Pound => {
                    let inner_error_reason = if just_parsed_doc_comment {
                        "an inner attribute is not permitted following an outer doc comment"
                    } else if !attrs.is_empty() {
                        "an inner attribute is not permitted following an outer attribute"
                    } else {
                        DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG
                    };
                    let inner_parse_policy =
                        InnerAttributeParsePolicy::NotPermitted { reason: inner_error_reason };
                    attrs.push(self.parse_attribute_with_inner_parse_policy(inner_parse_policy)?);
                    just_parsed_doc_comment = false;
                }
                token::DocComment(s) => {
                    let attr = ::attr::mk_sugared_doc_attr(
                        attr::mk_attr_id(),
                        self.id_to_interned_str(ast::Ident::with_empty_ctxt(s)),
                        self.span.lo,
                        self.span.hi
                    );
                    if attr.node.style != ast::AttrStyle::Outer {
                        let mut err = self.fatal("expected outer doc comment");
                        err.note("inner doc comments like this (starting with \
                                  `//!` or `/*!`) can only appear before items");
                        return Err(err);
                    }
                    attrs.push(attr);
                    self.bump();
                    just_parsed_doc_comment = true;
                }
                _ => break,
            }
        }
        return Ok(attrs);
    }

    /// Matches `attribute = # ! [ meta_item ]`
    ///
    /// If permit_inner is true, then a leading `!` indicates an inner
    /// attribute
    pub fn parse_attribute(&mut self, permit_inner: bool) -> PResult<'a, ast::Attribute> {
        debug!("parse_attribute: permit_inner={:?} self.token={:?}",
               permit_inner,
               self.token);
        let inner_parse_policy = if permit_inner {
            InnerAttributeParsePolicy::Permitted
        } else {
            InnerAttributeParsePolicy::NotPermitted
                { reason: DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG }
        };
        self.parse_attribute_with_inner_parse_policy(inner_parse_policy)
    }

    /// The same as `parse_attribute`, except it takes in an `InnerAttributeParsePolicy`
    /// that prescribes how to handle inner attributes.
    fn parse_attribute_with_inner_parse_policy(&mut self,
                                               inner_parse_policy: InnerAttributeParsePolicy)
                                               -> PResult<'a, ast::Attribute> {
        debug!("parse_attribute_with_inner_parse_policy: inner_parse_policy={:?} self.token={:?}",
               inner_parse_policy,
               self.token);
        let (span, value, mut style) = match self.token {
            token::Pound => {
                let lo = self.span.lo;
                self.bump();

                if inner_parse_policy == InnerAttributeParsePolicy::Permitted {
                    self.expected_tokens.push(TokenType::Token(token::Not));
                }
                let style = if self.token == token::Not {
                    self.bump();
                    if let InnerAttributeParsePolicy::NotPermitted { reason } = inner_parse_policy
                    {
                        let span = self.span;
                        self.diagnostic()
                            .struct_span_err(span, reason)
                            .note("inner attributes and doc comments, like `#![no_std]` or \
                                   `//! My crate`, annotate the item enclosing them, and are \
                                   usually found at the beginning of source files. Outer \
                                   attributes and doc comments, like `#[test]` and
                                   `/// My function`, annotate the item following them.")
                            .emit()
                    }
                    ast::AttrStyle::Inner
                } else {
                    ast::AttrStyle::Outer
                };

                self.expect(&token::OpenDelim(token::Bracket))?;
                let meta_item = self.parse_meta_item()?;
                let hi = self.span.hi;
                self.expect(&token::CloseDelim(token::Bracket))?;

                (mk_sp(lo, hi), meta_item, style)
            }
            _ => {
                let token_str = self.this_token_to_string();
                return Err(self.fatal(&format!("expected `#`, found `{}`", token_str)));
            }
        };

        if inner_parse_policy == InnerAttributeParsePolicy::Permitted &&
           self.token == token::Semi {
            self.bump();
            self.span_warn(span,
                           "this inner attribute syntax is deprecated. The new syntax is \
                            `#![foo]`, with a bang and no semicolon");
            style = ast::AttrStyle::Inner;
        }

        Ok(Spanned {
            span: span,
            node: ast::Attribute_ {
                id: attr::mk_attr_id(),
                style: style,
                value: value,
                is_sugared_doc: false,
            },
        })
    }

    /// Parse attributes that appear after the opening of an item. These should
    /// be preceded by an exclamation mark, but we accept and warn about one
    /// terminated by a semicolon.

    /// matches inner_attrs*
    pub fn parse_inner_attributes(&mut self) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = vec![];
        loop {
            match self.token {
                token::Pound => {
                    // Don't even try to parse if it's not an inner attribute.
                    if !self.look_ahead(1, |t| t == &token::Not) {
                        break;
                    }

                    let attr = self.parse_attribute(true)?;
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
                        self.bump();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        Ok(attrs)
    }

    /// matches meta_item = IDENT
    /// | IDENT = lit
    /// | IDENT meta_seq
    pub fn parse_meta_item(&mut self) -> PResult<'a, P<ast::MetaItem>> {
        let nt_meta = match self.token {
            token::Interpolated(token::NtMeta(ref e)) => Some(e.clone()),
            _ => None,
        };

        if let Some(meta) = nt_meta {
            self.bump();
            return Ok(meta);
        }

        let lo = self.span.lo;
        let ident = self.parse_ident()?;
        let name = self.id_to_interned_str(ident);
        match self.token {
            token::Eq => {
                self.bump();
                let lit = self.parse_lit()?;
                // FIXME #623 Non-string meta items are not serialized correctly;
                // just forbid them for now
                match lit.node {
                    ast::LitKind::Str(..) => {}
                    _ => {
                        self.span_err(lit.span,
                                      "non-string literals are not allowed in meta-items");
                    }
                }
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaItemKind::NameValue(name, lit))))
            }
            token::OpenDelim(token::Paren) => {
                let inner_items = self.parse_meta_seq()?;
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaItemKind::List(name, inner_items))))
            }
            _ => {
                let hi = self.last_span.hi;
                Ok(P(spanned(lo, hi, ast::MetaItemKind::Word(name))))
            }
        }
    }

    /// matches meta_seq = ( COMMASEP(meta_item) )
    fn parse_meta_seq(&mut self) -> PResult<'a, Vec<P<ast::MetaItem>>> {
        self.parse_unspanned_seq(&token::OpenDelim(token::Paren),
                                 &token::CloseDelim(token::Paren),
                                 SeqSep::trailing_allowed(token::Comma),
                                 |p: &mut Parser<'a>| p.parse_meta_item())
    }
}
