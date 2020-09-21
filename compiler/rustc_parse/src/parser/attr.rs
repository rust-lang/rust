use super::{FlatToken, Parser, PathStyle};
use rustc_ast as ast;
use rustc_ast::attr::{self, HasAttrs};
use rustc_ast::token::{self, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{
    AttributesData, DelimSpan, PreexpTokenStream, PreexpTokenTree, Spacing, TokenStream,
};
use rustc_ast::{AttrVec, Attribute};
use rustc_ast_pretty::pprust;
use rustc_errors::{error_code, Handler, PResult};
use rustc_span::symbol::sym;
use rustc_span::{Span, DUMMY_SP};

use tracing::debug;

#[derive(Debug)]
pub(super) enum InnerAttrPolicy<'a> {
    Permitted,
    Forbidden { reason: &'a str, saw_doc_comment: bool, prev_attr_sp: Option<Span> },
}

const DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG: &str = "an inner attribute is not \
                                                     permitted in this context";

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum SupportsCustomAttr {
    Yes,
    No,
}

pub struct CfgAttrItem {
    pub item: ast::AttrItem,
    pub span: Span,
    pub tokens: TokenStream,
}

pub(super) const DEFAULT_INNER_ATTR_FORBIDDEN: InnerAttrPolicy<'_> = InnerAttrPolicy::Forbidden {
    reason: DEFAULT_UNEXPECTED_INNER_ATTR_ERR_MSG,
    saw_doc_comment: false,
    prev_attr_sp: None,
};

impl<'a> Parser<'a> {
    fn has_any_attributes(&mut self) -> bool {
        self.check(&token::Pound) || matches!(self.token.kind, token::DocComment(..))
    }

    fn parse_outer_attributes_(
        &mut self,
        custom: SupportsCustomAttr,
    ) -> PResult<'a, Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        let mut just_parsed_doc_comment = false;

        loop {
            let mut parse_attr = |this: &mut Self| {
                debug!("parse_outer_attributes: self.token={:?}", this.token);
                if this.check(&token::Pound) {
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
                    let attr = this.parse_attribute_with_inner_parse_policy(inner_parse_policy)?;
                    just_parsed_doc_comment = false;
                    Ok((Some(attr), Vec::new())) // Attributes don't have their own attributes
                } else if let token::DocComment(comment_kind, attr_style, data) = this.token.kind {
                    let attr =
                        attr::mk_doc_comment(comment_kind, attr_style, data, this.token.span);
                    if attr.style != ast::AttrStyle::Outer {
                        this.sess
                            .span_diagnostic
                            .struct_span_err_with_code(
                                this.token.span,
                                "expected outer doc comment",
                                error_code!(E0753),
                            )
                            .note(
                                "inner doc comments like this (starting with \
                                 `//!` or `/*!`) can only appear before items",
                            )
                            .emit();
                    }
                    this.bump();
                    just_parsed_doc_comment = true;
                    Ok((Some(attr), Vec::new()))
                } else {
                    Ok((None, Vec::new()))
                }
            };

            // `in_derive` does not take into account the attributes we are currently parsing
            // (which may contain a `derive`). This is fine - if a `derive` attribute
            // can legally occur here, `custom` will be `SupportsCustomAttr::Yes`
            let (attr, tokens) = if custom == SupportsCustomAttr::Yes || self.in_derive {
                let (attr, tokens) = self.collect_tokens_keep_in_stream(false, parse_attr)?;
                (attr, Some(tokens))
            } else {
                let (attr, _nested_attrs) = parse_attr(self)?;
                (attr, None)
            };

            if let Some(mut attr) = attr {
                attr.tokens = tokens.map(|t| t.to_tokenstream());
                attrs.push(attr);
            } else {
                break;
            }
        }
        Ok(attrs)
    }

    pub(super) fn parse_or_use_outer_attributes<
        R: HasAttrs,
        F: FnOnce(&mut Self, AttrVec) -> PResult<'a, R>,
    >(
        &mut self,
        already_parsed_attrs: Option<AttrVec>,
        custom: SupportsCustomAttr,
        f: F,
    ) -> PResult<'a, (R, Option<PreexpTokenStream>)> {
        let in_derive = self.in_derive;
        let needs_tokens = |attrs: &[Attribute]| attrs_require_tokens(in_derive, attrs);

        let make_capture_res = |this: &mut Self, f: F, attrs: AttrVec| {
            let (res, tokens) = this.collect_tokens(|this| {
                let mut new_attrs = attrs.clone().to_vec();

                let old_in_derive = this.in_derive;
                this.in_derive =
                    old_in_derive || new_attrs.iter().any(|attr| attr.has_name(sym::derive));
                let res = f(this, attrs);
                this.in_derive = old_in_derive;

                let mut res = res?;

                // `this.in_derive` does not take into account our new attributes
                // (which may contain a `derive`). This is fine - if a `derive` attribute
                // can legally occur here, `custom` will be `SupportsCustomAttr::Yes`
                if custom == SupportsCustomAttr::Yes || this.in_derive {
                    res.visit_attrs(|attrs| {
                        new_attrs = attrs.clone();
                    });
                    Ok((res, new_attrs))
                } else {
                    Ok((res, Vec::new()))
                }
            })?;
            Ok((res, Some(tokens)))
        };

        if let Some(attrs) = already_parsed_attrs {
            if needs_tokens(&attrs) {
                return make_capture_res(self, f, attrs);
            } else {
                return f(self, attrs).map(|res| (res, None));
            }
        } else {
            // If we are already collecting tokens, we need to
            // perform token collection here even if we have no
            // outer attributes, since there may be inner attributes
            // parsed by 'f'.
            if !self.has_any_attributes() && !self.in_derive {
                return Ok((f(self, AttrVec::new())?, None));
            }

            let attrs = self.parse_outer_attributes_(custom)?;
            if !needs_tokens(&attrs) {
                return Ok((f(self, attrs.into())?, None));
            }

            return make_capture_res(self, f, attrs.into());
        }
    }

    pub(super) fn parse_outer_attributes<R: HasAttrs>(
        &mut self,
        custom: SupportsCustomAttr,
        f: impl FnOnce(&mut Self, Vec<ast::Attribute>) -> PResult<'a, R>,
    ) -> PResult<'a, R> {
        self.parse_outer_attributes_with_tokens(custom, f).map(|(res, _tokens)| res)
    }

    /// Parses attributes that appear before an item.
    pub(super) fn parse_outer_attributes_with_tokens<R: HasAttrs>(
        &mut self,
        custom: SupportsCustomAttr,
        f: impl FnOnce(&mut Self, Vec<ast::Attribute>) -> PResult<'a, R>,
    ) -> PResult<'a, (R, Option<PreexpTokenStream>)> {
        self.parse_or_use_outer_attributes(None, custom, |this, attrs| f(this, attrs.into()))
    }

    /// Matches `attribute = # ! [ meta_item ]`.
    ///
    /// If `permit_inner` is `true`, then a leading `!` indicates an inner
    /// attribute.
    pub fn parse_attribute(&mut self, permit_inner: bool) -> PResult<'a, ast::Attribute> {
        debug!("parse_attribute: permit_inner={:?} self.token={:?}", permit_inner, self.token);
        let inner_parse_policy =
            if permit_inner { InnerAttrPolicy::Permitted } else { DEFAULT_INNER_ATTR_FORBIDDEN };
        self.parse_attribute_with_inner_parse_policy(inner_parse_policy)
    }

    /// The same as `parse_attribute`, except it takes in an `InnerAttrPolicy`
    /// that prescribes how to handle inner attributes.
    fn parse_attribute_with_inner_parse_policy(
        &mut self,
        inner_parse_policy: InnerAttrPolicy<'_>,
    ) -> PResult<'a, ast::Attribute> {
        debug!(
            "parse_attribute_with_inner_parse_policy: inner_parse_policy={:?} self.token={:?}",
            inner_parse_policy, self.token
        );
        let lo = self.token.span;
        let (span, item, style) = if self.eat(&token::Pound) {
            let style =
                if self.eat(&token::Not) { ast::AttrStyle::Inner } else { ast::AttrStyle::Outer };

            self.expect(&token::OpenDelim(token::Bracket))?;
            let item = self.parse_attr_item()?;
            self.expect(&token::CloseDelim(token::Bracket))?;
            let attr_sp = lo.to(self.prev_token.span);

            // Emit error if inner attribute is encountered and forbidden.
            if style == ast::AttrStyle::Inner {
                self.error_on_forbidden_inner_attr(attr_sp, inner_parse_policy);
            }

            (attr_sp, item, style)
        } else {
            let token_str = pprust::token_to_string(&self.token);
            let msg = &format!("expected `#`, found `{}`", token_str);
            return Err(self.struct_span_err(self.token.span, msg));
        };

        Ok(attr::mk_attr_from_item(style, item, span))
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
            ast::AttrItem { path, args, tokens: None }
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
            let (attr, tokens) = self.collect_tokens_no_attrs(|this| {
                // Only try to parse if it is an inner attribute (has `!`).
                if this.check(&token::Pound) && this.look_ahead(1, |t| t == &token::Not) {
                    let attr = this.parse_attribute(true)?;
                    assert_eq!(attr.style, ast::AttrStyle::Inner);
                    Ok(Some(attr))
                } else if let token::DocComment(comment_kind, attr_style, data) = this.token.kind {
                    // We need to get the position of this token before we bump.
                    let attr =
                        attr::mk_doc_comment(comment_kind, attr_style, data, this.token.span);
                    if attr.style == ast::AttrStyle::Inner {
                        this.bump();
                        Ok(Some(attr))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            })?;
            if let Some(mut attr) = attr {
                attr.tokens = Some(tokens.to_tokenstream());
                attrs.push(attr)
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
    pub fn parse_cfg_attr(&mut self) -> PResult<'a, (ast::MetaItem, Vec<CfgAttrItem>)> {
        let cfg_predicate = self.parse_meta_item()?;
        self.expect(&token::Comma)?;

        // Presumably, the majority of the time there will only be one attr.
        let mut expanded_attrs = Vec::with_capacity(1);
        while self.token.kind != token::Eof {
            let lo = self.token.span;
            let (item, tokens) =
                self.collect_tokens(|this| this.parse_attr_item().map(|item| (item, Vec::new())))?;
            expanded_attrs.push(CfgAttrItem {
                item,
                span: lo.to(self.prev_token.span),
                tokens: tokens.to_tokenstream(),
            });

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

    pub(super) fn collect_tokens_keep_in_stream<R>(
        &mut self,
        keep_in_stream: bool,
        f: impl FnOnce(&mut Self) -> PResult<'a, (R, Vec<ast::Attribute>)>,
    ) -> PResult<'a, (R, PreexpTokenStream)> {
        let start_pos = self.token_cursor.collecting_buf.len() - 1;
        let prev_collecting = std::mem::replace(&mut self.token_cursor.is_collecting, true);

        let ret = f(self);

        let err_stream = if ret.is_err() {
            // Rustdoc tries to parse an item, and then cancels the error
            // if it fails.
            // FIXME: Come up with a better way of doing this
            if !self.is_rustdoc {
                self.sess
                    .span_diagnostic
                    .delay_span_bug(self.token.span, "Parse error during token collection");
            }
            Some(PreexpTokenStream::new(vec![]))
        } else {
            None
        };

        fn make_stream(
            handler: &Handler,
            iter: impl Iterator<Item = (FlatToken, Spacing)>,
            err_stream: Option<PreexpTokenStream>,
        ) -> PreexpTokenStream {
            err_stream.unwrap_or_else(|| make_preexp_stream(handler, iter))
        }

        let last_token = self.token_cursor.collecting_buf.pop().unwrap();
        let mut stream = if prev_collecting {
            if keep_in_stream {
                make_stream(
                    &self.sess.span_diagnostic,
                    self.token_cursor.collecting_buf[start_pos..].iter().cloned(),
                    err_stream,
                )
            } else {
                make_stream(
                    &self.sess.span_diagnostic,
                    self.token_cursor.collecting_buf.drain(start_pos..),
                    err_stream,
                )
            }
        } else {
            debug_assert_eq!(start_pos, 0);
            make_stream(
                &self.sess.span_diagnostic,
                std::mem::take(&mut self.token_cursor.collecting_buf).into_iter(),
                err_stream,
            )
        };

        if let Ok((_, attrs)) = ret.as_ref() {
            if !attrs.is_empty() {
                let data = AttributesData { attrs: attrs.clone(), tokens: stream };
                let tree = (PreexpTokenTree::OuterAttributes(data.clone()), Spacing::Alone);
                stream = PreexpTokenStream::new(vec![tree]);

                if prev_collecting {
                    assert!(keep_in_stream);
                    self.token_cursor.collecting_buf.splice(
                        start_pos..,
                        std::iter::once((FlatToken::OuterAttributes(data), Spacing::Alone)),
                    );
                }
            }
        }

        self.token_cursor.collecting_buf.push(last_token);
        self.token_cursor.is_collecting = prev_collecting;

        Ok((ret?.0, stream))
    }
}

fn make_preexp_stream(
    handler: &Handler,
    tokens: impl Iterator<Item = (FlatToken, Spacing)>,
) -> PreexpTokenStream {
    #[derive(Debug)]
    struct FrameData {
        open: Span,
        inner: Vec<(PreexpTokenTree, Spacing)>,
    }
    let mut stack = vec![FrameData { open: DUMMY_SP, inner: vec![] }];
    for tree in tokens {
        match tree.0 {
            FlatToken::Token(Token { kind: TokenKind::OpenDelim(_), span }) => {
                stack.push(FrameData { open: span, inner: vec![] });
            }
            FlatToken::Token(Token { kind: TokenKind::CloseDelim(delim), span }) => {
                let frame_data = stack.pop().expect("Token stack was empty!");
                let dspan = DelimSpan::from_pair(frame_data.open, span);
                let stream = PreexpTokenStream::new(frame_data.inner);
                let delimited = PreexpTokenTree::Delimited(dspan, delim, stream);
                stack
                    .last_mut()
                    .expect("Bottom token frame is missing!")
                    .inner
                    .push((delimited, Spacing::Alone));
            }
            FlatToken::Token(token) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push((PreexpTokenTree::Token(token), tree.1)),
            FlatToken::OuterAttributes(data) => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push((PreexpTokenTree::OuterAttributes(data), Spacing::Alone)),
        }
    }
    let final_buf = stack.pop().expect("Missing final buf!");
    if !stack.is_empty() {
        handler.delay_span_bug(
            stack[0].open,
            &format!("Stack should be empty: final_buf={:?} stack={:?}", final_buf, stack),
        );
    }
    PreexpTokenStream::new(final_buf.inner)
}

pub fn attrs_require_tokens(in_derive: bool, attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if let Some(ident) = attr.ident() {
            ident.name == sym::derive
                // We only need tokens for 'cfgs' inside a derive,
                // since cfg-stripping occurs before derive expansion
                || (ident.name == sym::cfg && in_derive)
                // This might apply a custom attribute/derive
                || ident.name == sym::cfg_attr
                || !rustc_feature::is_builtin_attr_name(ident.name)
        } else {
            true
        }
    })
}
