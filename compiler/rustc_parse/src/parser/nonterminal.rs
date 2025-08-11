use rustc_ast::token::NtExprKind::*;
use rustc_ast::token::NtPatKind::*;
use rustc_ast::token::{self, InvisibleOrigin, MetaVarKind, NonterminalKind, Token};
use rustc_errors::PResult;
use rustc_span::{Ident, kw};

use crate::errors::UnexpectedNonterminal;
use crate::parser::pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
use crate::parser::{FollowedByType, ForceCollect, ParseNtResult, Parser, PathStyle};

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
                | MetaVarKind::Literal // `true`, `false`
                | MetaVarKind::Meta { .. }
                | MetaVarKind::Path => true,

                MetaVarKind::Item
                | MetaVarKind::Block
                | MetaVarKind::Vis => false,

                MetaVarKind::Ident
                | MetaVarKind::Lifetime
                | MetaVarKind::TT => unreachable!(),
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
                    | MetaVarKind::Vis => false,
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
            NonterminalKind::TT | NonterminalKind::Item | NonterminalKind::Stmt => {
                token.kind.close_delim().is_none()
            }
        }
    }

    /// Parse a non-terminal (e.g. MBE `:pat` or `:ident`). Inlined because there is only one call
    /// site.
    #[inline]
    pub fn parse_nonterminal(&mut self, kind: NonterminalKind) -> PResult<'a, ParseNtResult> {
        // A `macro_rules!` invocation may pass a captured item/expr to a proc-macro,
        // which requires having captured tokens available. Since we cannot determine
        // in advance whether or not a proc-macro will be (transitively) invoked,
        // we always capture tokens for any nonterminal that needs them.
        match kind {
            // Note that TT is treated differently to all the others.
            NonterminalKind::TT => Ok(ParseNtResult::Tt(self.parse_token_tree())),
            NonterminalKind::Item => match self.parse_item(ForceCollect::Yes)? {
                Some(item) => Ok(ParseNtResult::Item(item)),
                None => Err(self.dcx().create_err(UnexpectedNonterminal::Item(self.token.span))),
            },
            NonterminalKind::Block => {
                // While a block *expression* may have attributes (e.g. `#[my_attr] { ... }`),
                // the ':block' matcher does not support them
                Ok(ParseNtResult::Block(self.collect_tokens_no_attrs(|this| this.parse_block())?))
            }
            NonterminalKind::Stmt => match self.parse_stmt(ForceCollect::Yes)? {
                Some(stmt) => Ok(ParseNtResult::Stmt(Box::new(stmt))),
                None => {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Statement(self.token.span)))
                }
            },
            NonterminalKind::Pat(pat_kind) => Ok(ParseNtResult::Pat(
                self.collect_tokens_no_attrs(|this| match pat_kind {
                    PatParam { .. } => this.parse_pat_no_top_alt(None, None),
                    PatWithOr => this.parse_pat_no_top_guard(
                        None,
                        RecoverComma::No,
                        RecoverColon::No,
                        CommaRecoveryMode::EitherTupleOrPipe,
                    ),
                })?,
                pat_kind,
            )),
            NonterminalKind::Expr(expr_kind) => {
                Ok(ParseNtResult::Expr(self.parse_expr_force_collect()?, expr_kind))
            }
            NonterminalKind::Literal => {
                // The `:literal` matcher does not support attributes.
                Ok(ParseNtResult::Literal(
                    self.collect_tokens_no_attrs(|this| this.parse_literal_maybe_minus())?,
                ))
            }
            NonterminalKind::Ty => Ok(ParseNtResult::Ty(
                self.collect_tokens_no_attrs(|this| this.parse_ty_no_question_mark_recover())?,
            )),
            // This could be handled like a token, since it is one.
            NonterminalKind::Ident => {
                if let Some((ident, is_raw)) = get_macro_ident(&self.token) {
                    self.bump();
                    Ok(ParseNtResult::Ident(ident, is_raw))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Ident {
                        span: self.token.span,
                        token: self.token,
                    }))
                }
            }
            NonterminalKind::Path => Ok(ParseNtResult::Path(Box::new(
                self.collect_tokens_no_attrs(|this| this.parse_path(PathStyle::Type))?,
            ))),
            NonterminalKind::Meta => {
                Ok(ParseNtResult::Meta(Box::new(self.parse_attr_item(ForceCollect::Yes)?)))
            }
            NonterminalKind::Vis => Ok(ParseNtResult::Vis(Box::new(
                self.collect_tokens_no_attrs(|this| this.parse_visibility(FollowedByType::Yes))?,
            ))),
            NonterminalKind::Lifetime => {
                // We want to keep `'keyword` parsing, just like `keyword` is still
                // an ident for nonterminal purposes.
                if let Some((ident, is_raw)) = self.token.lifetime() {
                    self.bump();
                    Ok(ParseNtResult::Lifetime(ident, is_raw))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Lifetime {
                        span: self.token.span,
                        token: self.token,
                    }))
                }
            }
        }
    }
}

/// The token is an identifier, but not `_`.
/// We prohibit passing `_` to macros expecting `ident` for now.
fn get_macro_ident(token: &Token) -> Option<(Ident, token::IdentIsRaw)> {
    token.ident().filter(|(ident, _)| ident.name != kw::Underscore)
}
