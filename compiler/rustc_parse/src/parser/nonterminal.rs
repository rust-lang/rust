use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, NonterminalKind, Token};
use rustc_ast::HasTokens;
use rustc_ast_pretty::pprust;
use rustc_errors::IntoDiagnostic;
use rustc_errors::PResult;
use rustc_span::symbol::{kw, Ident};

use crate::errors::UnexpectedNonterminal;
use crate::parser::pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
use crate::parser::{FollowedByType, ForceCollect, NtOrTt, Parser, PathStyle};

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
            match *nt {
                token::NtItem(_) | token::NtBlock(_) | token::NtVis(_) | token::NtLifetime(_) => {
                    false
                }
                _ => true,
            }
        }

        match kind {
            NonterminalKind::Expr => {
                token.can_begin_expr()
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Let)
                // This exception is here for backwards compatibility.
                && !token.is_keyword(kw::Const)
            }
            NonterminalKind::Ty => token.can_begin_type(),
            NonterminalKind::Ident => get_macro_ident(token).is_some(),
            NonterminalKind::Literal => token.can_begin_literal_maybe_minus(),
            NonterminalKind::Vis => match token.kind {
                // The follow-set of :vis + "priv" keyword + interpolated
                token::Comma | token::Ident(..) | token::Interpolated(..) => true,
                _ => token.can_begin_type(),
            },
            NonterminalKind::Block => match &token.kind {
                token::OpenDelim(Delimiter::Brace) => true,
                token::Interpolated(nt) => !matches!(
                    **nt,
                    token::NtItem(_)
                        | token::NtPat(_)
                        | token::NtTy(_)
                        | token::NtIdent(..)
                        | token::NtMeta(_)
                        | token::NtPath(_)
                        | token::NtVis(_)
                ),
                _ => false,
            },
            NonterminalKind::Path | NonterminalKind::Meta => match &token.kind {
                token::ModSep | token::Ident(..) => true,
                token::Interpolated(nt) => match **nt {
                    token::NtPath(_) | token::NtMeta(_) => true,
                    _ => may_be_ident(&nt),
                },
                _ => false,
            },
            NonterminalKind::PatParam { .. } | NonterminalKind::PatWithOr { .. } => {
                match &token.kind {
                token::Ident(..) |                          // box, ref, mut, and other identifiers (can stricten)
                token::OpenDelim(Delimiter::Parenthesis) |  // tuple pattern
                token::OpenDelim(Delimiter::Bracket) |      // slice pattern
                token::BinOp(token::And) |                  // reference
                token::BinOp(token::Minus) |                // negative literal
                token::AndAnd |                             // double reference
                token::Literal(..) |                        // literal
                token::DotDot |                             // range pattern (future compat)
                token::DotDotDot |                          // range pattern (future compat)
                token::ModSep |                             // path
                token::Lt |                                 // path (UFCS constant)
                token::BinOp(token::Shl) => true,           // path (double UFCS)
                // leading vert `|` or-pattern
                token::BinOp(token::Or) =>  matches!(kind, NonterminalKind::PatWithOr {..}),
                token::Interpolated(nt) => may_be_ident(nt),
                _ => false,
            }
            }
            NonterminalKind::Lifetime => match &token.kind {
                token::Lifetime(_) => true,
                token::Interpolated(nt) => {
                    matches!(**nt, token::NtLifetime(_))
                }
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
    pub fn parse_nonterminal(&mut self, kind: NonterminalKind) -> PResult<'a, NtOrTt> {
        // Any `Nonterminal` which stores its tokens (currently `NtItem` and `NtExpr`)
        // needs to have them force-captured here.
        // A `macro_rules!` invocation may pass a captured item/expr to a proc-macro,
        // which requires having captured tokens available. Since we cannot determine
        // in advance whether or not a proc-macro will be (transitively) invoked,
        // we always capture tokens for any `Nonterminal` which needs them.
        let mut nt = match kind {
            // Note that TT is treated differently to all the others.
            NonterminalKind::TT => return Ok(NtOrTt::Tt(self.parse_token_tree())),
            NonterminalKind::Item => match self.parse_item(ForceCollect::Yes)? {
                Some(item) => token::NtItem(item),
                None => {
                    return Err(UnexpectedNonterminal::Item(self.token.span)
                               .into_diagnostic(&self.sess.span_diagnostic));
                }
            },
            NonterminalKind::Block => {
                // While a block *expression* may have attributes (e.g. `#[my_attr] { ... }`),
                // the ':block' matcher does not support them
                token::NtBlock(self.collect_tokens_no_attrs(|this| this.parse_block())?)
            }
            NonterminalKind::Stmt => match self.parse_stmt(ForceCollect::Yes)? {
                Some(s) => token::NtStmt(P(s)),
                None => {
                    return Err(UnexpectedNonterminal::Statement(self.token.span)
                               .into_diagnostic(&self.sess.span_diagnostic));
                }
            },
            NonterminalKind::PatParam { .. } | NonterminalKind::PatWithOr { .. } => {
                token::NtPat(self.collect_tokens_no_attrs(|this| match kind {
                    NonterminalKind::PatParam { .. } => this.parse_pat_no_top_alt(None),
                    NonterminalKind::PatWithOr { .. } => this.parse_pat_allow_top_alt(
                        None,
                        RecoverComma::No,
                        RecoverColon::No,
                        CommaRecoveryMode::EitherTupleOrPipe,
                    ),
                    _ => unreachable!(),
                })?)
            }

            NonterminalKind::Expr => token::NtExpr(self.parse_expr_force_collect()?),
            NonterminalKind::Literal => {
                // The `:literal` matcher does not support attributes
                token::NtLiteral(
                    self.collect_tokens_no_attrs(|this| this.parse_literal_maybe_minus())?,
                )
            }

            NonterminalKind::Ty => token::NtTy(
                self.collect_tokens_no_attrs(|this| this.parse_no_question_mark_recover())?,
            ),

            // this could be handled like a token, since it is one
            NonterminalKind::Ident
                if let Some((ident, is_raw)) = get_macro_ident(&self.token) =>
            {
                self.bump();
                token::NtIdent(ident, is_raw)
            }
            NonterminalKind::Ident => {
                return Err(UnexpectedNonterminal::Ident {
                    span: self.token.span,
                    token: self.token.clone(),
                }.into_diagnostic(&self.sess.span_diagnostic));
            }
            NonterminalKind::Path => token::NtPath(
                P(self.collect_tokens_no_attrs(|this| this.parse_path(PathStyle::Type))?),
            ),
            NonterminalKind::Meta => token::NtMeta(P(self.parse_attr_item(true)?)),
            NonterminalKind::Vis => token::NtVis(
                P(self.collect_tokens_no_attrs(|this| this.parse_visibility(FollowedByType::Yes))?),
            ),
            NonterminalKind::Lifetime => {
                if self.check_lifetime() {
                    token::NtLifetime(self.expect_lifetime().ident)
                } else {
                    return Err(UnexpectedNonterminal::Lifetime {
                        span: self.token.span,
                        token: self.token.clone(),
                    }.into_diagnostic(&self.sess.span_diagnostic));
                }
            }
        };

        // If tokens are supported at all, they should be collected.
        if matches!(nt.tokens_mut(), Some(None)) {
            panic!(
                "Missing tokens for nt {:?} at {:?}: {:?}",
                nt,
                nt.span(),
                pprust::nonterminal_to_string(&nt)
            );
        }

        Ok(NtOrTt::Nt(nt))
    }
}

/// The token is an identifier, but not `_`.
/// We prohibit passing `_` to macros expecting `ident` for now.
fn get_macro_ident(token: &Token) -> Option<(Ident, bool)> {
    token.ident().filter(|(ident, _)| ident.name != kw::Underscore)
}
