use super::{Parser, PathStyle, TokenType};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_span::{Span, Symbol};
use syntax::ast;
use syntax::attr;
use syntax::token::{self, Nonterminal};
use syntax::util::comments;

use log::debug;

#[derive(Debug)]
enum InnerAttributeParsePolicy<'a> {
    Permitted,
    NotPermitted { reason: &'a str, saw_doc_comment: bool, prev_attr_sp: Option<Span> },
}

const DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG: &str = "an inner attribute is not \
                                                     permitted in this context";

impl<'a> Parser<'a> {
    /// Parses attributes that appear before an item.
    pub(super) fn parse_outer_attributes(&mut self) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        let mut just_parsed_doc_comment = false;
        loop {
            debug!("parse_outer_attributes: self.token={:?}", self.token);
            match self.token.kind {
                token::Pound => {
                    let inner_error_reason = if just_parsed_doc_comment {
                        "an inner attribute is not permitted following an outer doc comment"
                    } else if !attrs.is_empty() {
                        "an inner attribute is not permitted following an outer attribute"
                    } else {
                        DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG
                    };
                    let inner_parse_policy = InnerAttributeParsePolicy::NotPermitted {
                        reason: inner_error_reason,
                        saw_doc_comment: just_parsed_doc_comment,
                        prev_attr_sp: attrs.last().and_then(|a| Some(a.span)),
                    };
                    let attr = self.parse_attribute_with_inner_parse_policy(inner_parse_policy)?;
                    attrs.push(attr);
                    just_parsed_doc_comment = false;
                }
                token::DocComment(s) => {
                    let attr = self.mk_doc_comment(s);
                    if attr.style != ast::AttrStyle::Outer {
                        let span = self.token.span;
                        let mut err = self.struct_span_err(span, "expected outer doc comment");
                        err.note(
                            "inner doc comments like this (starting with \
                                  `//!` or `/*!`) can only appear before items",
                        );
                        return Err(err);
                    }
                    attrs.push(attr);
                    self.bump();
                    just_parsed_doc_comment = true;
                }
                _ => break,
            }
        }
        Ok(attrs)
    }

    fn mk_doc_comment(&self, s: Symbol) -> ast::Attribute {
        let style = comments::doc_comment_style(&s.as_str());
        attr::mk_doc_comment(style, s, self.token.span)
    }

    /// Matches `attribute = # ! [ meta_item ]`.
    ///
    /// If `permit_inner` is `true`, then a leading `!` indicates an inner
    /// attribute.
    pub fn parse_attribute(&mut self, permit_inner: bool) -> PResult<'a, ast::Attribute> {
        debug!("parse_attribute: permit_inner={:?} self.token={:?}", permit_inner, self.token);
        let inner_parse_policy = if permit_inner {
            InnerAttributeParsePolicy::Permitted
        } else {
            InnerAttributeParsePolicy::NotPermitted {
                reason: DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG,
                saw_doc_comment: false,
                prev_attr_sp: None,
            }
        };
        self.parse_attribute_with_inner_parse_policy(inner_parse_policy)
    }

    /// The same as `parse_attribute`, except it takes in an `InnerAttributeParsePolicy`
    /// that prescribes how to handle inner attributes.
    fn parse_attribute_with_inner_parse_policy(
        &mut self,
        inner_parse_policy: InnerAttributeParsePolicy<'_>,
    ) -> PResult<'a, ast::Attribute> {
        debug!(
            "parse_attribute_with_inner_parse_policy: inner_parse_policy={:?} self.token={:?}",
            inner_parse_policy, self.token
        );
        let (span, item, style) = match self.token.kind {
            token::Pound => {
                let lo = self.token.span;
                self.bump();

                if let InnerAttributeParsePolicy::Permitted = inner_parse_policy {
                    self.expected_tokens.push(TokenType::Token(token::Not));
                }

                let style = if self.token == token::Not {
                    self.bump();
                    ast::AttrStyle::Inner
                } else {
                    ast::AttrStyle::Outer
                };

                self.expect(&token::OpenDelim(token::Bracket))?;
                let item = self.parse_attr_item()?;
                self.expect(&token::CloseDelim(token::Bracket))?;
                let hi = self.prev_span;

                let attr_sp = lo.to(hi);

                // Emit error if inner attribute is encountered and not permitted
                if style == ast::AttrStyle::Inner {
                    if let InnerAttributeParsePolicy::NotPermitted {
                        reason,
                        saw_doc_comment,
                        prev_attr_sp,
                    } = inner_parse_policy
                    {
                        let prev_attr_note = if saw_doc_comment {
                            "previous doc comment"
                        } else {
                            "previous outer attribute"
                        };

                        let mut diagnostic = self.struct_span_err(attr_sp, reason);

                        if let Some(prev_attr_sp) = prev_attr_sp {
                            diagnostic
                                .span_label(attr_sp, "not permitted following an outer attibute")
                                .span_label(prev_attr_sp, prev_attr_note);
                        }

                        diagnostic
                            .note(
                                "inner attributes, like `#![no_std]`, annotate the item \
                                   enclosing them, and are usually found at the beginning of \
                                   source files. Outer attributes, like `#[test]`, annotate the \
                                   item following them.",
                            )
                            .emit();
                    }
                }

                (attr_sp, item, style)
            }
            _ => {
                let token_str = pprust::token_to_string(&self.token);
                let msg = &format!("expected `#`, found `{}`", token_str);
                return Err(self.struct_span_err(self.token.span, msg));
            }
        };

        Ok(attr::mk_attr_from_item(style, item, span))
    }

    /// Parses an inner part of an attribute (the path and following tokens).
    /// The tokens must be either a delimited token stream, or empty token stream,
    /// or the "legacy" key-value form.
    ///     PATH `(` TOKEN_STREAM `)`
    ///     PATH `[` TOKEN_STREAM `]`
    ///     PATH `{` TOKEN_STREAM `}`
    ///     PATH
    ///     PATH `=` UNSUFFIXED_LIT
    /// The delimiters or `=` are still put into the resulting token stream.
    pub fn parse_attr_item(&mut self) -> PResult<'a, ast::AttrItem> {
        let item = match self.token.kind {
            token::Interpolated(ref nt) => match **nt {
                Nonterminal::NtMeta(ref item) => Some(item.clone().into_inner()),
                _ => None,
            },
            _ => None,
        };
        Ok(if let Some(item) = item {
            self.bump();
            item
        } else {
            let path = self.parse_path(PathStyle::Mod)?;
            let args = self.parse_attr_args()?;
            ast::AttrItem { path, args }
        })
    }

    /// Parses attributes that appear after the opening of an item. These should
    /// be preceded by an exclamation mark, but we accept and warn about one
    /// terminated by a semicolon.
    ///
    /// Matches `inner_attrs*`.
    crate fn parse_inner_attributes(&mut self) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = vec![];
        loop {
            match self.token.kind {
                token::Pound => {
                    // Don't even try to parse if it's not an inner attribute.
                    if !self.look_ahead(1, |t| t == &token::Not) {
                        break;
                    }

                    let attr = self.parse_attribute(true)?;
                    assert_eq!(attr.style, ast::AttrStyle::Inner);
                    attrs.push(attr);
                }
                token::DocComment(s) => {
                    // We need to get the position of this token before we bump.
                    let attr = self.mk_doc_comment(s);
                    if attr.style == ast::AttrStyle::Inner {
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

    crate fn parse_unsuffixed_lit(&mut self) -> PResult<'a, ast::Lit> {
        let lit = self.parse_lit()?;
        debug!("checking if {:?} is unusuffixed", lit);

        if !lit.kind.is_unsuffixed() {
            let msg = "suffixed literals are not allowed in attributes";
            self.struct_span_err(lit.span, msg)
                .help(
                    "instead of using a suffixed literal \
                                    (`1u8`, `1.0f32`, etc.), use an unsuffixed version \
                                    (`1`, `1.0`, etc.)",
                )
                .emit();
        }

        Ok(lit)
    }

    /// Parses `cfg_attr(pred, attr_item_list)` where `attr_item_list` is comma-delimited.
    pub fn parse_cfg_attr(&mut self) -> PResult<'a, (ast::MetaItem, Vec<(ast::AttrItem, Span)>)> {
        let cfg_predicate = self.parse_meta_item()?;
        self.expect(&token::Comma)?;

        // Presumably, the majority of the time there will only be one attr.
        let mut expanded_attrs = Vec::with_capacity(1);
        while self.token.kind != token::Eof {
            let lo = self.token.span;
            let item = self.parse_attr_item()?;
            expanded_attrs.push((item, lo.to(self.prev_span)));
            if !self.eat(&token::Comma) {
                break;
            }
        }

        Ok((cfg_predicate, expanded_attrs))
    }

    /// Matches `COMMASEP(meta_item_inner)`.
    crate fn parse_meta_seq_top(&mut self) -> PResult<'a, Vec<ast::NestedMetaItem>> {
        // Presumably, the majority of the time there will only be one attr.
        let mut nmis = Vec::with_capacity(1);
        while self.token.kind != token::Eof {
            nmis.push(self.parse_meta_item_inner()?);
            if !self.eat(&token::Comma) {
                break;
            }
        }
        Ok(nmis)
    }

    /// Matches the following grammar (per RFC 1559).
    ///
    ///     meta_item : PATH ( '=' UNSUFFIXED_LIT | '(' meta_item_inner? ')' )? ;
    ///     meta_item_inner : (meta_item | UNSUFFIXED_LIT) (',' meta_item_inner)? ;
    pub fn parse_meta_item(&mut self) -> PResult<'a, ast::MetaItem> {
        let nt_meta = match self.token.kind {
            token::Interpolated(ref nt) => match **nt {
                token::NtMeta(ref e) => Some(e.clone()),
                _ => None,
            },
            _ => None,
        };

        if let Some(item) = nt_meta {
            return match item.meta(item.path.span) {
                Some(meta) => {
                    self.bump();
                    Ok(meta)
                }
                None => self.unexpected(),
            };
        }

        let lo = self.token.span;
        let path = self.parse_path(PathStyle::Mod)?;
        let kind = self.parse_meta_item_kind()?;
        let span = lo.to(self.prev_span);
        Ok(ast::MetaItem { path, kind, span })
    }

    crate fn parse_meta_item_kind(&mut self) -> PResult<'a, ast::MetaItemKind> {
        Ok(if self.eat(&token::Eq) {
            ast::MetaItemKind::NameValue(self.parse_unsuffixed_lit()?)
        } else if self.check(&token::OpenDelim(token::Paren)) {
            // Matches `meta_seq = ( COMMASEP(meta_item_inner) )`.
            let (list, _) = self.parse_paren_comma_seq(|p| p.parse_meta_item_inner())?;
            ast::MetaItemKind::List(list)
        } else {
            ast::MetaItemKind::Word
        })
    }

    /// Matches `meta_item_inner : (meta_item | UNSUFFIXED_LIT) ;`.
    fn parse_meta_item_inner(&mut self) -> PResult<'a, ast::NestedMetaItem> {
        match self.parse_unsuffixed_lit() {
            Ok(lit) => return Ok(ast::NestedMetaItem::Literal(lit)),
            Err(ref mut err) => err.cancel(),
        }

        match self.parse_meta_item() {
            Ok(mi) => return Ok(ast::NestedMetaItem::MetaItem(mi)),
            Err(ref mut err) => err.cancel(),
        }

        let found = pprust::token_to_string(&self.token);
        let msg = format!("expected unsuffixed literal or identifier, found `{}`", found);
        Err(self.struct_span_err(self.token.span, &msg))
    }
}
