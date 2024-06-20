use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, Nonterminal::*, NonterminalKind, Token};
use rustc_ast::HasTokens;
use rustc_ast_pretty::pprust;
use rustc_data_structures::sync::Lrc;
use rustc_errors::PResult;
use rustc_span::symbol::{kw, Ident};

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
        fn may_be_ident(nt: &token::Nonterminal) -> bool {
            match nt {
                NtStmt(_)
                | NtPat(_)
                | NtExpr(_)
                | NtTy(_)
                | NtLiteral(_) // `true`, `false`
                | NtMeta(_)
                | NtPath(_) => true,

                NtItem(_)
                | NtBlock(_)
                | NtVis(_) => false,
            }
        }

        match kind {
            NonterminalKind::Expr2021 { inferred: _ } => {
                token.can_begin_expr()
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Let)
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Const)
            }
            NonterminalKind::Expr => {
                token.can_begin_expr()
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Let)
            }
            NonterminalKind::Ty => token.can_begin_type(),
            NonterminalKind::Ident => get_macro_ident(token).is_some(),
            NonterminalKind::Literal => token.can_begin_literal_maybe_minus(),
            NonterminalKind::Vis => match token.kind {
                // The follow-set of :vis + "priv" keyword + interpolated
                token::Comma
                | token::Ident(..)
                | token::NtIdent(..)
                | token::NtLifetime(..)
                | token::Interpolated(_) => true,
                _ => token.can_begin_type(),
            },
            NonterminalKind::Block => match &token.kind {
                token::OpenDelim(Delimiter::Brace) => true,
                token::NtLifetime(..) => true,
                token::Interpolated(nt) => match &**nt {
                    NtBlock(_) | NtStmt(_) | NtExpr(_) | NtLiteral(_) => true,
                    NtItem(_) | NtPat(_) | NtTy(_) | NtMeta(_) | NtPath(_) | NtVis(_) => false,
                },
                _ => false,
            },
            NonterminalKind::Path | NonterminalKind::Meta => match &token.kind {
                token::PathSep | token::Ident(..) | token::NtIdent(..) => true,
                token::Interpolated(nt) => may_be_ident(nt),
                _ => false,
            },
            NonterminalKind::PatParam { .. } | NonterminalKind::PatWithOr => match &token.kind {
                // box, ref, mut, and other identifiers (can stricten)
                token::Ident(..) | token::NtIdent(..) |
                token::OpenDelim(Delimiter::Parenthesis) |  // tuple pattern
                token::OpenDelim(Delimiter::Bracket) |      // slice pattern
                token::BinOp(token::And) |                  // reference
                token::BinOp(token::Minus) |                // negative literal
                token::AndAnd |                             // double reference
                token::Literal(_) |                         // literal
                token::DotDot |                             // range pattern (future compat)
                token::DotDotDot |                          // range pattern (future compat)
                token::PathSep |                             // path
                token::Lt |                                 // path (UFCS constant)
                token::BinOp(token::Shl) => true,           // path (double UFCS)
                // leading vert `|` or-pattern
                token::BinOp(token::Or) => matches!(kind, NonterminalKind::PatWithOr),
                token::Interpolated(nt) => may_be_ident(nt),
                _ => false,
            },
            NonterminalKind::Lifetime => match &token.kind {
                token::Lifetime(_) | token::NtLifetime(..) => true,
                _ => false,
            },
            NonterminalKind::TT | NonterminalKind::Item | NonterminalKind::Stmt => {
                !matches!(token.kind, token::CloseDelim(_))
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
        // we always capture tokens for any `Nonterminal` which needs them.
        let mut nt = match kind {
            // Note that TT is treated differently to all the others.
            NonterminalKind::TT => return Ok(ParseNtResult::Tt(self.parse_token_tree())),
            NonterminalKind::Item => match self.parse_item(ForceCollect::Yes)? {
                Some(item) => NtItem(item),
                None => {
                    return Err(self
                        .dcx()
                        .create_err(UnexpectedNonterminal::Item(self.token.span)));
                }
            },
            NonterminalKind::Block => {
                // While a block *expression* may have attributes (e.g. `#[my_attr] { ... }`),
                // the ':block' matcher does not support them
                NtBlock(self.collect_tokens_no_attrs(|this| this.parse_block())?)
            }
            NonterminalKind::Stmt => match self.parse_stmt(ForceCollect::Yes)? {
                Some(s) => NtStmt(P(s)),
                None => {
                    return Err(self
                        .dcx()
                        .create_err(UnexpectedNonterminal::Statement(self.token.span)));
                }
            },
            NonterminalKind::PatParam { .. } | NonterminalKind::PatWithOr => {
                NtPat(self.collect_tokens_no_attrs(|this| match kind {
                    NonterminalKind::PatParam { .. } => this.parse_pat_no_top_alt(None, None),
                    NonterminalKind::PatWithOr => this.parse_pat_allow_top_alt(
                        None,
                        RecoverComma::No,
                        RecoverColon::No,
                        CommaRecoveryMode::EitherTupleOrPipe,
                    ),
                    _ => unreachable!(),
                })?)
            }

            NonterminalKind::Expr | NonterminalKind::Expr2021 { inferred: _ } => {
                NtExpr(self.parse_expr_force_collect()?)
            }
            NonterminalKind::Literal => {
                // The `:literal` matcher does not support attributes
                NtLiteral(self.collect_tokens_no_attrs(|this| this.parse_literal_maybe_minus())?)
            }

            NonterminalKind::Ty => {
                NtTy(self.collect_tokens_no_attrs(|this| this.parse_ty_no_question_mark_recover())?)
            }

            // this could be handled like a token, since it is one
            NonterminalKind::Ident => {
                return if let Some((ident, is_raw)) = get_macro_ident(&self.token) {
                    self.bump();
                    Ok(ParseNtResult::Ident(ident, is_raw))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Ident {
                        span: self.token.span,
                        token: self.token.clone(),
                    }))
                };
            }
            NonterminalKind::Path => {
                NtPath(P(self.collect_tokens_no_attrs(|this| this.parse_path(PathStyle::Type))?))
            }
            NonterminalKind::Meta => NtMeta(P(self.parse_attr_item(true)?)),
            NonterminalKind::Vis => {
                NtVis(P(self
                    .collect_tokens_no_attrs(|this| this.parse_visibility(FollowedByType::Yes))?))
            }
            NonterminalKind::Lifetime => {
                return if self.check_lifetime() {
                    Ok(ParseNtResult::Lifetime(self.expect_lifetime().ident))
                } else {
                    Err(self.dcx().create_err(UnexpectedNonterminal::Lifetime {
                        span: self.token.span,
                        token: self.token.clone(),
                    }))
                };
            }
        };

        // If tokens are supported at all, they should be collected.
        if matches!(nt.tokens_mut(), Some(None)) {
            panic!(
                "Missing tokens for nt {:?} at {:?}: {:?}",
                nt,
                nt.use_span(),
                pprust::nonterminal_to_string(&nt)
            );
        }

        Ok(ParseNtResult::Nt(Lrc::new(nt)))
    }
}

/// The token is an identifier, but not `_`.
/// We prohibit passing `_` to macros expecting `ident` for now.
fn get_macro_ident(token: &Token) -> Option<(Ident, token::IdentIsRaw)> {
    token.ident().filter(|(ident, _)| ident.name != kw::Underscore)
}
