use rustc_ast as ast;
use rustc_ast::token::{self, MetaVarKind};
use rustc_ast::tokenstream::ParserRange;
use rustc_ast::{Attribute, attr};
use rustc_errors::codes::*;
use rustc_errors::{Diag, PResult};
use rustc_span::{BytePos, Span};
use thin_vec::ThinVec;
use tracing::debug;

use super::{
    AttrWrapper, Capturing, FnParseMode, ForceCollect, Parser, PathStyle, Trailing, UsePreAttrPos,
};
use crate::parser::FnContext;
use crate::{errors, exp, fluent_generated as fluent};

// Public for rustfmt usage
#[derive(Debug)]
pub enum InnerAttrPolicy {
    Permitted,
    Forbidden(Option<InnerAttrForbiddenReason>),
}

#[derive(Clone, Copy, Debug)]
pub enum InnerAttrForbiddenReason {
    InCodeBlock,
    AfterOuterDocComment { prev_doc_comment_span: Span },
    AfterOuterAttribute { prev_outer_attr_sp: Span },
}

enum OuterAttributeType {
    DocComment,
    DocBlockComment,
    Attribute,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AllowLeadingUnsafe {
    Yes,
    No,
}

impl<'a> Parser<'a> {
    /// Parses attributes that appear before an item.
    pub(super) fn parse_outer_attributes(&mut self) -> PResult<'a, AttrWrapper> {
        let mut outer_attrs = ast::AttrVec::new();
        let mut just_parsed_doc_comment = false;
        let start_pos = self.num_bump_calls;
        loop {
            let attr = if self.check(exp!(Pound)) {
                let prev_outer_attr_sp = outer_attrs.last().map(|attr: &Attribute| attr.span);

                let inner_error_reason = if just_parsed_doc_comment {
                    Some(InnerAttrForbiddenReason::AfterOuterDocComment {
                        prev_doc_comment_span: prev_outer_attr_sp.unwrap(),
                    })
                } else {
                    prev_outer_attr_sp.map(|prev_outer_attr_sp| {
                        InnerAttrForbiddenReason::AfterOuterAttribute { prev_outer_attr_sp }
                    })
                };
                let inner_parse_policy = InnerAttrPolicy::Forbidden(inner_error_reason);
                just_parsed_doc_comment = false;
                Some(self.parse_attribute(inner_parse_policy)?)
            } else if let token::DocComment(comment_kind, attr_style, data) = self.token.kind {
                if attr_style != ast::AttrStyle::Outer {
                    let span = self.token.span;
                    let mut err = self
                        .dcx()
                        .struct_span_err(span, fluent::parse_inner_doc_comment_not_permitted);
                    err.code(E0753);
                    if let Some(replacement_span) = self.annotate_following_item_if_applicable(
                        &mut err,
                        span,
                        match comment_kind {
                            token::CommentKind::Line => OuterAttributeType::DocComment,
                            token::CommentKind::Block => OuterAttributeType::DocBlockComment,
                        },
                        true,
                    ) {
                        err.note(fluent::parse_note);
                        err.span_suggestion_verbose(
                            replacement_span,
                            fluent::parse_suggestion,
                            "",
                            rustc_errors::Applicability::MachineApplicable,
                        );
                    }
                    err.emit();
                }
                self.bump();
                just_parsed_doc_comment = true;
                // Always make an outer attribute - this allows us to recover from a misplaced
                // inner attribute.
                Some(attr::mk_doc_comment(
                    &self.psess.attr_id_generator,
                    comment_kind,
                    ast::AttrStyle::Outer,
                    data,
                    self.prev_token.span,
                ))
            } else {
                None
            };

            if let Some(attr) = attr {
                if attr.style == ast::AttrStyle::Outer {
                    outer_attrs.push(attr);
                }
            } else {
                break;
            }
        }
        Ok(AttrWrapper::new(outer_attrs, start_pos))
    }

    /// Matches `attribute = # ! [ meta_item ]`.
    /// `inner_parse_policy` prescribes how to handle inner attributes.
    // Public for rustfmt usage.
    pub fn parse_attribute(
        &mut self,
        inner_parse_policy: InnerAttrPolicy,
    ) -> PResult<'a, ast::Attribute> {
        debug!(
            "parse_attribute: inner_parse_policy={:?} self.token={:?}",
            inner_parse_policy, self.token
        );
        let lo = self.token.span;
        // Attributes can't have attributes of their own [Editor's note: not with that attitude]
        self.collect_tokens_no_attrs(|this| {
            let pound_hi = this.token.span.hi();
            assert!(this.eat(exp!(Pound)), "parse_attribute called in non-attribute position");

            let not_lo = this.token.span.lo();
            let style =
                if this.eat(exp!(Bang)) { ast::AttrStyle::Inner } else { ast::AttrStyle::Outer };

            let mut bracket_res = this.expect(exp!(OpenBracket));
            // If `#!` is not followed by `[`
            if let Err(err) = &mut bracket_res
                && style == ast::AttrStyle::Inner
                && pound_hi == not_lo
            {
                err.note(
                    "the token sequence `#!` here looks like the start of \
                    a shebang interpreter directive but it is not",
                );
                err.help(
                    "if you meant this to be a shebang interpreter directive, \
                    move it to the very start of the file",
                );
            }
            bracket_res?;
            let item = this.parse_attr_item(ForceCollect::No)?;
            this.expect(exp!(CloseBracket))?;
            let attr_sp = lo.to(this.prev_token.span);

            // Emit error if inner attribute is encountered and forbidden.
            if style == ast::AttrStyle::Inner {
                this.error_on_forbidden_inner_attr(
                    attr_sp,
                    inner_parse_policy,
                    item.is_valid_for_outer_style(),
                );
            }

            Ok(attr::mk_attr_from_item(&self.psess.attr_id_generator, item, None, style, attr_sp))
        })
    }

    fn annotate_following_item_if_applicable(
        &self,
        err: &mut Diag<'_>,
        span: Span,
        attr_type: OuterAttributeType,
        suggest_to_outer: bool,
    ) -> Option<Span> {
        let mut snapshot = self.create_snapshot_for_diagnostic();
        let lo = span.lo()
            + BytePos(match attr_type {
                OuterAttributeType::Attribute => 1,
                _ => 2,
            });
        let hi = lo + BytePos(1);
        let replacement_span = span.with_lo(lo).with_hi(hi);
        if let OuterAttributeType::DocBlockComment | OuterAttributeType::DocComment = attr_type {
            snapshot.bump();
        }
        loop {
            // skip any other attributes, we want the item
            if snapshot.token == token::Pound {
                if let Err(err) = snapshot.parse_attribute(InnerAttrPolicy::Permitted) {
                    err.cancel();
                    return Some(replacement_span);
                }
            } else {
                break;
            }
        }
        match snapshot.parse_item_common(
            AttrWrapper::empty(),
            true,
            false,
            FnParseMode { req_name: |_| true, context: FnContext::Free, req_body: true },
            ForceCollect::No,
        ) {
            Ok(Some(item)) => {
                // FIXME(#100717)
                err.arg("item", item.kind.descr());
                err.span_label(item.span, fluent::parse_label_does_not_annotate_this);
                if suggest_to_outer {
                    err.span_suggestion_verbose(
                        replacement_span,
                        fluent::parse_sugg_change_inner_to_outer,
                        match attr_type {
                            OuterAttributeType::Attribute => "",
                            OuterAttributeType::DocBlockComment => "*",
                            OuterAttributeType::DocComment => "/",
                        },
                        rustc_errors::Applicability::MachineApplicable,
                    );
                }
                return None;
            }
            Err(item_err) => {
                item_err.cancel();
            }
            Ok(None) => {}
        }
        Some(replacement_span)
    }

    pub(super) fn error_on_forbidden_inner_attr(
        &self,
        attr_sp: Span,
        policy: InnerAttrPolicy,
        suggest_to_outer: bool,
    ) {
        if let InnerAttrPolicy::Forbidden(reason) = policy {
            let mut diag = match reason.as_ref().copied() {
                Some(InnerAttrForbiddenReason::AfterOuterDocComment { prev_doc_comment_span }) => {
                    self.dcx()
                        .struct_span_err(
                            attr_sp,
                            fluent::parse_inner_attr_not_permitted_after_outer_doc_comment,
                        )
                        .with_span_label(attr_sp, fluent::parse_label_attr)
                        .with_span_label(
                            prev_doc_comment_span,
                            fluent::parse_label_prev_doc_comment,
                        )
                }
                Some(InnerAttrForbiddenReason::AfterOuterAttribute { prev_outer_attr_sp }) => self
                    .dcx()
                    .struct_span_err(
                        attr_sp,
                        fluent::parse_inner_attr_not_permitted_after_outer_attr,
                    )
                    .with_span_label(attr_sp, fluent::parse_label_attr)
                    .with_span_label(prev_outer_attr_sp, fluent::parse_label_prev_attr),
                Some(InnerAttrForbiddenReason::InCodeBlock) | None => {
                    self.dcx().struct_span_err(attr_sp, fluent::parse_inner_attr_not_permitted)
                }
            };

            diag.note(fluent::parse_inner_attr_explanation);
            if self
                .annotate_following_item_if_applicable(
                    &mut diag,
                    attr_sp,
                    OuterAttributeType::Attribute,
                    suggest_to_outer,
                )
                .is_some()
            {
                diag.note(fluent::parse_outer_attr_explanation);
            };
            diag.emit();
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
    pub fn parse_attr_item(&mut self, force_collect: ForceCollect) -> PResult<'a, ast::AttrItem> {
        if let Some(item) = self.eat_metavar_seq_with_matcher(
            |mv_kind| matches!(mv_kind, MetaVarKind::Meta { .. }),
            |this| this.parse_attr_item(force_collect),
        ) {
            return Ok(item);
        }

        // Attr items don't have attributes.
        self.collect_tokens(None, AttrWrapper::empty(), force_collect, |this, _empty_attrs| {
            let is_unsafe = this.eat_keyword(exp!(Unsafe));
            let unsafety = if is_unsafe {
                let unsafe_span = this.prev_token.span;
                this.expect(exp!(OpenParen))?;
                ast::Safety::Unsafe(unsafe_span)
            } else {
                ast::Safety::Default
            };

            let path = this.parse_path(PathStyle::Mod)?;
            let args = this.parse_attr_args()?;
            if is_unsafe {
                this.expect(exp!(CloseParen))?;
            }
            Ok((
                ast::AttrItem { unsafety, path, args, tokens: None },
                Trailing::No,
                UsePreAttrPos::No,
            ))
        })
    }

    /// Parses attributes that appear after the opening of an item. These should
    /// be preceded by an exclamation mark, but we accept and warn about one
    /// terminated by a semicolon.
    ///
    /// Matches `inner_attrs*`.
    pub fn parse_inner_attributes(&mut self) -> PResult<'a, ast::AttrVec> {
        let mut attrs = ast::AttrVec::new();
        loop {
            let start_pos = self.num_bump_calls;
            // Only try to parse if it is an inner attribute (has `!`).
            let attr = if self.check(exp!(Pound)) && self.look_ahead(1, |t| t == &token::Bang) {
                Some(self.parse_attribute(InnerAttrPolicy::Permitted)?)
            } else if let token::DocComment(comment_kind, attr_style, data) = self.token.kind {
                if attr_style == ast::AttrStyle::Inner {
                    self.bump();
                    Some(attr::mk_doc_comment(
                        &self.psess.attr_id_generator,
                        comment_kind,
                        attr_style,
                        data,
                        self.prev_token.span,
                    ))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(attr) = attr {
                // If we are currently capturing tokens (i.e. we are within a call to
                // `Parser::collect_tokens`) record the token positions of this inner attribute,
                // for possible later processing in a `LazyAttrTokenStream`.
                if let Capturing::Yes = self.capture_state.capturing {
                    let end_pos = self.num_bump_calls;
                    let parser_range = ParserRange(start_pos..end_pos);
                    self.capture_state.inner_attr_parser_ranges.insert(attr.id, parser_range);
                }
                attrs.push(attr);
            } else {
                break;
            }
        }
        Ok(attrs)
    }

    // Note: must be unsuffixed.
    pub(crate) fn parse_unsuffixed_meta_item_lit(&mut self) -> PResult<'a, ast::MetaItemLit> {
        let lit = self.parse_meta_item_lit()?;
        debug!("checking if {:?} is unsuffixed", lit);

        if !lit.kind.is_unsuffixed() {
            self.dcx().emit_err(errors::SuffixedLiteralInAttribute { span: lit.span });
        }

        Ok(lit)
    }

    /// Parses `cfg_attr(pred, attr_item_list)` where `attr_item_list` is comma-delimited.
    pub fn parse_cfg_attr(
        &mut self,
    ) -> PResult<'a, (ast::MetaItemInner, Vec<(ast::AttrItem, Span)>)> {
        let cfg_predicate = self.parse_meta_item_inner()?;
        self.expect(exp!(Comma))?;

        // Presumably, the majority of the time there will only be one attr.
        let mut expanded_attrs = Vec::with_capacity(1);
        while self.token != token::Eof {
            let lo = self.token.span;
            let item = self.parse_attr_item(ForceCollect::Yes)?;
            expanded_attrs.push((item, lo.to(self.prev_token.span)));
            if !self.eat(exp!(Comma)) {
                break;
            }
        }

        Ok((cfg_predicate, expanded_attrs))
    }

    /// Matches `COMMASEP(meta_item_inner)`.
    pub fn parse_meta_seq_top(&mut self) -> PResult<'a, ThinVec<ast::MetaItemInner>> {
        // Presumably, the majority of the time there will only be one attr.
        let mut nmis = ThinVec::with_capacity(1);
        while self.token != token::Eof {
            nmis.push(self.parse_meta_item_inner()?);
            if !self.eat(exp!(Comma)) {
                break;
            }
        }
        Ok(nmis)
    }

    /// Parse a meta item per RFC 1559.
    ///
    /// ```ebnf
    /// MetaItem = SimplePath ( '=' UNSUFFIXED_LIT | '(' MetaSeq? ')' )? ;
    /// MetaSeq = MetaItemInner (',' MetaItemInner)* ','? ;
    /// ```
    pub fn parse_meta_item(
        &mut self,
        unsafe_allowed: AllowLeadingUnsafe,
    ) -> PResult<'a, ast::MetaItem> {
        if let Some(MetaVarKind::Meta { has_meta_form }) = self.token.is_metavar_seq() {
            return if has_meta_form {
                let attr_item = self
                    .eat_metavar_seq(MetaVarKind::Meta { has_meta_form: true }, |this| {
                        this.parse_attr_item(ForceCollect::No)
                    })
                    .unwrap();
                Ok(attr_item.meta(attr_item.path.span).unwrap())
            } else {
                self.unexpected_any()
            };
        }

        let lo = self.token.span;
        let is_unsafe = if unsafe_allowed == AllowLeadingUnsafe::Yes {
            self.eat_keyword(exp!(Unsafe))
        } else {
            false
        };
        let unsafety = if is_unsafe {
            let unsafe_span = self.prev_token.span;
            self.expect(exp!(OpenParen))?;

            ast::Safety::Unsafe(unsafe_span)
        } else {
            ast::Safety::Default
        };

        let path = self.parse_path(PathStyle::Mod)?;
        let kind = self.parse_meta_item_kind()?;
        if is_unsafe {
            self.expect(exp!(CloseParen))?;
        }
        let span = lo.to(self.prev_token.span);

        Ok(ast::MetaItem { unsafety, path, kind, span })
    }

    pub(crate) fn parse_meta_item_kind(&mut self) -> PResult<'a, ast::MetaItemKind> {
        Ok(if self.eat(exp!(Eq)) {
            ast::MetaItemKind::NameValue(self.parse_unsuffixed_meta_item_lit()?)
        } else if self.check(exp!(OpenParen)) {
            let (list, _) = self.parse_paren_comma_seq(|p| p.parse_meta_item_inner())?;
            ast::MetaItemKind::List(list)
        } else {
            ast::MetaItemKind::Word
        })
    }

    /// Parse an inner meta item per RFC 1559.
    ///
    /// ```ebnf
    /// MetaItemInner = UNSUFFIXED_LIT | MetaItem ;
    /// ```
    pub fn parse_meta_item_inner(&mut self) -> PResult<'a, ast::MetaItemInner> {
        match self.parse_unsuffixed_meta_item_lit() {
            Ok(lit) => return Ok(ast::MetaItemInner::Lit(lit)),
            Err(err) => err.cancel(), // we provide a better error below
        }

        match self.parse_meta_item(AllowLeadingUnsafe::No) {
            Ok(mi) => return Ok(ast::MetaItemInner::MetaItem(mi)),
            Err(err) => err.cancel(), // we provide a better error below
        }

        let mut err = errors::InvalidMetaItem {
            span: self.token.span,
            descr: super::token_descr(&self.token),
            quote_ident_sugg: None,
        };

        // Suggest quoting idents, e.g. in `#[cfg(key = value)]`. We don't use `Token::ident` and
        // don't `uninterpolate` the token to avoid suggesting anything butchered or questionable
        // when macro metavariables are involved.
        if self.prev_token == token::Eq
            && let token::Ident(..) = self.token.kind
        {
            let before = self.token.span.shrink_to_lo();
            while let token::Ident(..) = self.token.kind {
                self.bump();
            }
            err.quote_ident_sugg = Some(errors::InvalidMetaItemQuoteIdentSugg {
                before,
                after: self.prev_token.span.shrink_to_hi(),
            });
        }

        Err(self.dcx().create_err(err))
    }
}
