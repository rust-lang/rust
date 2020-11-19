use super::{Parser, PathStyle};
use rustc_ast as ast;
use rustc_ast::attr;
use rustc_ast::token::{self, Nonterminal};
use rustc_ast_pretty::pprust;
use rustc_errors::{error_code, PResult};
use rustc_span::{sym, Span};

use tracing::debug;

#[derive(Debug)]
pub(super) enum InnerAttrPolicy<'a> {
    Permitted,
    Forbidden { reason: &'a str, saw_doc_comment: bool, prev_attr_sp: Option<Span> },
}

const DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG: &str = "an inner attribute is not \
                                                     permitted in this context";

pub(super) const DEFAULT_INNER_ATTR_FORBIDDEN: InnerAttrPolicy<'_> = InnerAttrPolicy::Forbidden {
    reason: DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG,
    saw_doc_comment: false,
    prev_attr_sp: None,
};

impl<'a> Parser<'a> {
    /// Parses attributes that appear before an item.
    pub(super) fn parse_outer_attributes(&mut self) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        let mut just_parsed_doc_comment = false;
        loop {
            debug!("parse_outer_attributes: self.token={:?}", self.token);
            let attr = if self.check(&token::Pound) {
                let inner_error_reason = if just_parsed_doc_comment {
                    "an inner attribute is not permitted following an outer doc comment"
                } else if !attrs.is_empty() {
                    "an inner attribute is not permitted following an outer attribute"
                } else {
                    DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG
                };
                let inner_parse_policy = InnerAttrPolicy::Forbidden {
                    reason: inner_error_reason,
                    saw_doc_comment: just_parsed_doc_comment,
                    prev_attr_sp: attrs.last().map(|a| a.span),
                };
                just_parsed_doc_comment = false;
                Some(self.parse_attribute(inner_parse_policy)?)
            } else if let token::DocComment(comment_kind, attr_style, data) = self.token.kind {
                if attr_style != ast::AttrStyle::Outer {
                    self.sess
                        .span_diagnostic
                        .struct_span_err_with_code(
                            self.token.span,
                            "expected outer doc comment",
                            error_code!(E0753),
                        )
                        .note(
                            "inner doc comments like this (starting with \
                         `//!` or `/*!`) can only appear before items",
                        )
                        .emit();
                }
                self.bump();
                just_parsed_doc_comment = true;
                Some(attr::mk_doc_comment(comment_kind, attr_style, data, self.prev_token.span))
            } else {
                None
            };

            if let Some(attr) = attr {
                attrs.push(attr);
            } else {
                break;
            }
        }
        Ok(attrs)
    }

    /// Matches `attribute = # ! [ meta_item ]`.
    /// `inner_parse_policy` prescribes how to handle inner attributes.
    fn parse_attribute(
        &mut self,
        inner_parse_policy: InnerAttrPolicy<'_>,
    ) -> PResult<'a, ast::Attribute> {
        debug!(
            "parse_attribute: inner_parse_policy={:?} self.token={:?}",
            inner_parse_policy, self.token
        );
        let lo = self.token.span;
        let ((item, style, span), tokens) = self.collect_tokens(|this| {
            if this.eat(&token::Pound) {
                let style = if this.eat(&token::Not) {
                    ast::AttrStyle::Inner
                } else {
                    ast::AttrStyle::Outer
                };

                this.expect(&token::OpenDelim(token::Bracket))?;
                let item = this.parse_attr_item(false)?;
                this.expect(&token::CloseDelim(token::Bracket))?;
                let attr_sp = lo.to(this.prev_token.span);

                // Emit error if inner attribute is encountered and forbidden.
                if style == ast::AttrStyle::Inner {
                    this.error_on_forbidden_inner_attr(attr_sp, inner_parse_policy);
                }

                Ok((item, style, attr_sp))
            } else {
                let token_str = pprust::token_to_string(&this.token);
                let msg = &format!("expected `#`, found `{}`", token_str);
                Err(this.struct_span_err(this.token.span, msg))
            }
        })?;

        Ok(attr::mk_attr_from_item(item, tokens, style, span))
    }

    pub(super) fn error_on_forbidden_inner_attr(&self, attr_sp: Span, policy: InnerAttrPolicy<'_>) {
        if let InnerAttrPolicy::Forbidden { reason, saw_doc_comment, prev_attr_sp } = policy {
            let prev_attr_note =
                if saw_doc_comment { "previous doc comment" } else { "previous outer attribute" };

            let mut diag = self.struct_span_err(attr_sp, reason);

            if let Some(prev_attr_sp) = prev_attr_sp {
                diag.span_label(attr_sp, "not permitted following an outer attribute")
                    .span_label(prev_attr_sp, prev_attr_note);
            }

            diag.note(
                "inner attributes, like `#![no_std]`, annotate the item enclosing them, \
                and are usually found at the beginning of source files. \
                Outer attributes, like `#[test]`, annotate the item following them.",
            )
            .emit();
        }
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
    pub fn parse_attr_item(&mut self, capture_tokens: bool) -> PResult<'a, ast::AttrItem> {
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
            let do_parse = |this: &mut Self| {
                let path = this.parse_path(PathStyle::Mod)?;
                let args = this.parse_attr_args()?;
                Ok(ast::AttrItem { path, args, tokens: None })
            };
            if capture_tokens {
                let (mut item, tokens) = self.collect_tokens(do_parse)?;
                item.tokens = tokens;
                item
            } else {
                do_parse(self)?
            }
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
            // Only try to parse if it is an inner attribute (has `!`).
            let attr = if self.check(&token::Pound) && self.look_ahead(1, |t| t == &token::Not) {
                Some(self.parse_attribute(InnerAttrPolicy::Permitted)?)
            } else if let token::DocComment(comment_kind, attr_style, data) = self.token.kind {
                if attr_style == ast::AttrStyle::Inner {
                    self.bump();
                    Some(attr::mk_doc_comment(comment_kind, attr_style, data, self.prev_token.span))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(attr) = attr {
                attrs.push(attr);
            } else {
                break;
            }
        }
        Ok(attrs)
    }

    crate fn parse_unsuffixed_lit(&mut self) -> PResult<'a, ast::Lit> {
        let lit = self.parse_lit()?;
        debug!("checking if {:?} is unusuffixed", lit);

        if !lit.kind.is_unsuffixed() {
            self.struct_span_err(lit.span, "suffixed literals are not allowed in attributes")
                .help(
                    "instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), \
                    use an unsuffixed version (`1`, `1.0`, etc.)",
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
            let item = self.parse_attr_item(true)?;
            expanded_attrs.push((item, lo.to(self.prev_token.span)));
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
        let span = lo.to(self.prev_token.span);
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

pub fn maybe_needs_tokens(attrs: &[ast::Attribute]) -> bool {
    // One of the attributes may either itself be a macro, or apply derive macros (`derive`),
    // or expand to macro attributes (`cfg_attr`).
    attrs.iter().any(|attr| {
        attr.ident().map_or(true, |ident| {
            ident.name == sym::derive
                || ident.name == sym::cfg_attr
                || !rustc_feature::is_builtin_attr_name(ident.name)
        })
    })
}
