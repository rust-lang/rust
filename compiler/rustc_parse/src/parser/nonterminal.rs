use rustc_ast::token::NtExprKind::*;
use rustc_ast::token::NtPatKind::*;
use rustc_ast::token::{self, IdentIsRaw, InvisibleOrigin, MetaVarKind, NonterminalKind, Token};
use rustc_ast::tokenstream::{TokenStream, WithTokens};
use rustc_ast::{ExprKind, StmtKind, TyKind, UnOp};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_span::{BytePos, Ident, kw};

use crate::diagnostics::UnexpectedNonterminal;
use crate::parser::pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
use crate::parser::{
    AllowConstBlockItems, FollowedByType, ForceCollect, ParseNtResult, Parser, PathStyle,
};

impl<'a> Parser<'a> {
    /// Checks whether a non-terminal may begin with a particular token.
    ///
    /// Returning `false` is a *stability guarantee* that such a matcher will *never* begin with
    /// that token. Be conservative (return true) if not sure. Inlined because it has a single call
    /// site.
    #[inline]
    pub fn nonterminal_may_begin_with(kind: NonterminalKind, token: &Token) -> bool {
        /// Checks whether the non-terminal may contain a single (non-keyword) identifier.
        fn may_be_ident(kind: MetaVarKind) -> bool {
            match kind {
                MetaVarKind::Stmt
                | MetaVarKind::Pat(_)
                | MetaVarKind::Expr { .. }
                | MetaVarKind::Ty { .. }
                | MetaVarKind::Meta { .. }
                | MetaVarKind::Path => true,
                // `true`, `false`
                MetaVarKind::Literal => true,

                MetaVarKind::Item | MetaVarKind::Block | MetaVarKind::Vis | MetaVarKind::Guard => {
                    false
                }

                MetaVarKind::Ident | MetaVarKind::Lifetime | MetaVarKind::TT => unreachable!(),
            }
        }

        match kind {
            // `expr_2021` and earlier
            NonterminalKind::Expr(Expr2021 { .. }) => {
                token.can_begin_expr()
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Let)
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Const)
            }
            // Current edition expressions
            NonterminalKind::Expr(Expr) => {
                // In Edition 2024, `_` is considered an expression, so we
                // need to allow it here because `token.can_begin_expr()` does
                // not consider `_` to be an expression.
                //
                // Because `can_begin_expr` is used elsewhere, we need to reduce
                // the scope of where the `_` is considered an expression to
                // just macro parsing code.
                (token.can_begin_expr() || token.is_keyword(kw::Underscore))
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Let)
            }
            NonterminalKind::Ty => token.can_begin_type(),
            NonterminalKind::Ident => get_macro_ident(token).is_some(),
            NonterminalKind::Literal => token.can_begin_literal_maybe_minus(),
            NonterminalKind::Vis => match token.kind {
                // The follow-set of :vis + "priv" keyword + interpolated/metavar-expansion.
                token::Comma
                | token::Ident(..)
                | token::NtIdent(..)
                | token::NtLifetime(..)
                | token::OpenInvisible(InvisibleOrigin::MetaVar(_)) => true,
                _ => token.can_begin_type(),
            },
            NonterminalKind::Block => match &token.kind {
                token::OpenBrace => true,
                token::NtLifetime(..) => true,
                token::OpenInvisible(InvisibleOrigin::MetaVar(k)) => match k {
                    MetaVarKind::Block
                    | MetaVarKind::Stmt
                    | MetaVarKind::Expr { .. }
                    | MetaVarKind::Literal => true,
                    MetaVarKind::Item
                    | MetaVarKind::Pat(_)
                    | MetaVarKind::Ty { .. }
                    | MetaVarKind::Meta { .. }
                    | MetaVarKind::Path
                    | MetaVarKind::Vis
                    | MetaVarKind::Guard => false,
                    MetaVarKind::Lifetime | MetaVarKind::Ident | MetaVarKind::TT => {
                        unreachable!()
                    }
                },
                _ => false,
            },
            NonterminalKind::Path | NonterminalKind::Meta => match &token.kind {
                token::PathSep | token::Ident(..) | token::NtIdent(..) => true,
                token::OpenInvisible(InvisibleOrigin::MetaVar(kind)) => may_be_ident(*kind),
                _ => false,
            },
            NonterminalKind::Pat(pat_kind) => token.can_begin_pattern(pat_kind),
            NonterminalKind::Lifetime => match &token.kind {
                token::Lifetime(..) | token::NtLifetime(..) => true,
                _ => false,
            },
            NonterminalKind::Guard => match token.kind {
                token::OpenInvisible(InvisibleOrigin::MetaVar(MetaVarKind::Guard)) => true,
                _ => token.is_keyword(kw::If),
            },
            NonterminalKind::TT | NonterminalKind::Item | NonterminalKind::Stmt => {
                token.kind != token::Eof && token.kind.close_delim().is_none()
            }
        }
    }

    /// Parse a non-terminal (e.g. MBE `:pat` or `:ident`). Inlined because there is only one call
    /// site.
    #[inline]
    pub fn parse_nonterminal(&mut self, kind: NonterminalKind) -> PResult<'a, ParseNtResult> {
        // Extra information is collected to help re-parse the meta-variable later.
        // FIXME: Don't build a real AST; we just need a token stream.
        match kind {
            // Note that TT is treated differently to all the others.
            NonterminalKind::TT => Ok(ParseNtResult::Tt(self.parse_token_tree())),
            NonterminalKind::Item => {
                let item =
                    self.parse_item(ForceCollect::Yes, AllowConstBlockItems::Yes)?.ok_or_else(
                        || self.dcx().create_err(UnexpectedNonterminal::Item(self.token.span)),
                    )?;
                Ok(ParseNtResult::Item(item.span, TokenStream::from_ast(&item)))
            }
            NonterminalKind::Block => {
                // While a block *expression* may have attributes (e.g. `#[my_attr] { ... }`),
                // the ':block' matcher does not support them
                let block =
                    self.collect_tokens_no_attrs(|this| this.parse_block().map(WithTokens::new))?;
                Ok(ParseNtResult::Block(block.node.span, TokenStream::from_ast(&block)))
            }
            NonterminalKind::Stmt => {
                let stmt = self.parse_stmt(ForceCollect::Yes)?.ok_or_else(|| {
                    self.dcx().create_err(UnexpectedNonterminal::Statement(self.token.span))
                })?;
                let stream = if let StmtKind::Empty = stmt.kind {
                    // FIXME: Properly collect tokens for empty statements.
                    TokenStream::token_alone(token::Semi, stmt.span)
                } else {
                    TokenStream::from_ast(&stmt)
                };
                Ok(ParseNtResult::Stmt(stmt.span, stream))
            }
            NonterminalKind::Pat(pat_kind) => {
                let pat = self.collect_tokens_no_attrs(|this| {
                    match pat_kind {
                        PatParam { .. } => this.parse_pat_no_top_alt(None, None),
                        PatWithOr => this.parse_pat_no_top_guard(
                            None,
                            RecoverComma::No,
                            RecoverColon::No,
                            CommaRecoveryMode::EitherTupleOrPipe,
                        ),
                    }
                    .map(|pat| WithTokens::new(Box::new(pat)))
                })?;
                Ok(ParseNtResult::Pat(pat.node.span, TokenStream::from_ast(&pat), pat_kind))
            }
            NonterminalKind::Expr(expr_kind) => {
                let expr = self.parse_expr_force_collect()?;
                let (can_begin_literal_maybe_minus, can_begin_string_literal) = match &expr.kind {
                    ExprKind::Lit(_) => (true, true),
                    ExprKind::Unary(UnOp::Neg, e) if matches!(&e.kind, ExprKind::Lit(_)) => {
                        (true, false)
                    }
                    _ => (false, false),
                };

                Ok(ParseNtResult::Expr {
                    span: expr.span,
                    tokens: TokenStream::from_ast(&expr),
                    kind: expr_kind,
                    can_begin_literal_maybe_minus,
                    can_begin_string_literal,
                })
            }
            NonterminalKind::Literal => {
                // The `:literal` matcher does not support attributes.
                let lit = self.collect_tokens_no_attrs(|this| this.parse_literal_maybe_minus())?;
                Ok(ParseNtResult::Literal(lit.span, TokenStream::from_ast(&lit)))
            }
            NonterminalKind::Ty => {
                let ty = self.collect_tokens_no_attrs(|this| {
                    this.parse_ty_no_question_mark_recover().map(WithTokens::new)
                })?;
                let is_path = matches!(&ty.node.kind, TyKind::Path(None, _path));
                Ok(ParseNtResult::Ty {
                    span: ty.node.span,
                    tokens: TokenStream::from_ast(&ty),
                    is_path,
                })
            }
            // This could be handled like a token, since it is one.
            NonterminalKind::Ident => {
                if let Some((ident, is_raw)) = get_macro_ident(&self.token) {
                    self.bump();
                    Ok(ParseNtResult::Ident(ident, is_raw))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Ident {
                        span: self.token.span,
                        token: pprust::token_to_string(&self.token),
                    }))
                }
            }
            NonterminalKind::Path => {
                let path = self.collect_tokens_no_attrs(|this| {
                    this.parse_path(PathStyle::Type).map(WithTokens::new)
                })?;
                Ok(ParseNtResult::Path(path.node.span, TokenStream::from_ast(&path)))
            }
            NonterminalKind::Meta => {
                let attr_item = self.parse_attr_item(ForceCollect::Yes)?;
                let has_meta_form = attr_item.node.meta_kind().is_some();
                Ok(ParseNtResult::Meta {
                    span: attr_item.node.span(),
                    tokens: TokenStream::from_ast(&attr_item),
                    has_meta_form,
                })
            }
            NonterminalKind::Vis => {
                let vis = self.collect_tokens_no_attrs(|this| {
                    this.parse_visibility(FollowedByType::Yes).map(|vis| WithTokens::new(vis))
                })?;
                Ok(ParseNtResult::Vis(vis.node.span, TokenStream::from_ast(&vis)))
            }
            NonterminalKind::Lifetime => {
                // We want to keep `'keyword` parsing, just like `keyword` is still
                // an ident for nonterminal purposes.
                if let Some((ident, is_raw)) = self.token.lifetime() {
                    self.bump();
                    Ok(ParseNtResult::Lifetime(ident, is_raw))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Lifetime {
                        span: self.token.span,
                        token: pprust::token_to_string(&self.token),
                    }))
                }
            }
            NonterminalKind::Guard => {
                let guard = self.expect_match_arm_guard(ForceCollect::Yes)?;

                // FIXME(macro_guard_matcher):
                // Perhaps it would be better to treat the leading `if` as part of `ast::Guard` during parsing?
                // Currently they are separate, but in macros we match and emit the leading `if` for `:guard` matchers, which creates some inconsistency.

                let leading_if_span = guard
                    .span_with_leading_if
                    .with_hi(guard.span_with_leading_if.lo() + BytePos(2));
                let mut ts =
                    TokenStream::token_alone(token::Ident(kw::If, IdentIsRaw::No), leading_if_span);
                ts.push_stream(TokenStream::from_ast(&guard.cond));

                Ok(ParseNtResult::Guard(guard.span_with_leading_if, ts))
            }
        }
    }
}

/// The token is an identifier, but not `_`.
/// We prohibit passing `_` to macros expecting `ident` for now.
fn get_macro_ident(token: &Token) -> Option<(Ident, token::IdentIsRaw)> {
    token.ident().filter(|(ident, _)| ident.name != kw::Underscore)
}
