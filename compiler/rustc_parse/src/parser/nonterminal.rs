use rustc_ast::ptr::P;
use rustc_ast::token::{self, Nonterminal, NonterminalKind, Token};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_span::symbol::{kw, Ident};

use crate::parser::pat::{GateOr, RecoverComma};
use crate::parser::{FollowedByType, Parser, PathStyle};

impl<'a> Parser<'a> {
    /// Checks whether a non-terminal may begin with a particular token.
    ///
    /// Returning `false` is a *stability guarantee* that such a matcher will *never* begin with that
    /// token. Be conservative (return true) if not sure.
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
            NonterminalKind::Block => match token.kind {
                token::OpenDelim(token::Brace) => true,
                token::Interpolated(ref nt) => !matches!(**nt, token::NtItem(_)
                    | token::NtPat(_)
                    | token::NtTy(_)
                    | token::NtIdent(..)
                    | token::NtMeta(_)
                    | token::NtPath(_)
                    | token::NtVis(_)),
                _ => false,
            },
            NonterminalKind::Path | NonterminalKind::Meta => match token.kind {
                token::ModSep | token::Ident(..) => true,
                token::Interpolated(ref nt) => match **nt {
                    token::NtPath(_) | token::NtMeta(_) => true,
                    _ => may_be_ident(&nt),
                },
                _ => false,
            },
            NonterminalKind::Pat2018 { .. } | NonterminalKind::Pat2021 { .. } => match token.kind {
                token::Ident(..) |                  // box, ref, mut, and other identifiers (can stricten)
                token::OpenDelim(token::Paren) |    // tuple pattern
                token::OpenDelim(token::Bracket) |  // slice pattern
                token::BinOp(token::And) |          // reference
                token::BinOp(token::Minus) |        // negative literal
                token::AndAnd |                     // double reference
                token::Literal(..) |                // literal
                token::DotDot |                     // range pattern (future compat)
                token::DotDotDot |                  // range pattern (future compat)
                token::ModSep |                     // path
                token::Lt |                         // path (UFCS constant)
                token::BinOp(token::Shl) => true,   // path (double UFCS)
                // leading vert `|` or-pattern
                token::BinOp(token::Or) =>  matches!(kind, NonterminalKind::Pat2021 {..}),
                token::Interpolated(ref nt) => may_be_ident(nt),
                _ => false,
            },
            NonterminalKind::Lifetime => match token.kind {
                token::Lifetime(_) => true,
                token::Interpolated(ref nt) => {
                    matches!(**nt, token::NtLifetime(_) | token::NtTT(_))
                }
                _ => false,
            },
            NonterminalKind::TT | NonterminalKind::Item | NonterminalKind::Stmt => {
                !matches!(token.kind, token::CloseDelim(_))
            }
        }
    }

    /// Parse a non-terminal (e.g. MBE `:pat` or `:ident`).
    pub fn parse_nonterminal(&mut self, kind: NonterminalKind) -> PResult<'a, Nonterminal> {
        // Any `Nonterminal` which stores its tokens (currently `NtItem` and `NtExpr`)
        // needs to have them force-captured here.
        // A `macro_rules!` invocation may pass a captured item/expr to a proc-macro,
        // which requires having captured tokens available. Since we cannot determine
        // in advance whether or not a proc-macro will be (transitively) invoked,
        // we always capture tokens for any `Nonterminal` which needs them.
        Ok(match kind {
            NonterminalKind::Item => match self.collect_tokens(|this| this.parse_item())? {
                Some(item) => token::NtItem(item),
                None => {
                    return Err(self.struct_span_err(self.token.span, "expected an item keyword"));
                }
            },
            NonterminalKind::Block => {
                token::NtBlock(self.collect_tokens(|this| this.parse_block())?)
            }
            NonterminalKind::Stmt => match self.collect_tokens(|this| this.parse_stmt())? {
                Some(s) => token::NtStmt(s),
                None => {
                    return Err(self.struct_span_err(self.token.span, "expected a statement"));
                }
            },
            NonterminalKind::Pat2018 { .. } | NonterminalKind::Pat2021 { .. } => {
                token::NtPat(self.collect_tokens(|this| match kind {
                    NonterminalKind::Pat2018 { .. } => this.parse_pat(None),
                    NonterminalKind::Pat2021 { .. } => {
                        this.parse_top_pat(GateOr::Yes, RecoverComma::No)
                    }
                    _ => unreachable!(),
                })?)
            }
            NonterminalKind::Expr => token::NtExpr(self.collect_tokens(|this| this.parse_expr())?),
            NonterminalKind::Literal => {
                token::NtLiteral(self.collect_tokens(|this| this.parse_literal_maybe_minus())?)
            }
            NonterminalKind::Ty => token::NtTy(self.collect_tokens(|this| this.parse_ty())?),
            // this could be handled like a token, since it is one
            NonterminalKind::Ident => {
                if let Some((ident, is_raw)) = get_macro_ident(&self.token) {
                    self.bump();
                    token::NtIdent(ident, is_raw)
                } else {
                    let token_str = pprust::token_to_string(&self.token);
                    let msg = &format!("expected ident, found {}", &token_str);
                    return Err(self.struct_span_err(self.token.span, msg));
                }
            }
            NonterminalKind::Path => {
                token::NtPath(self.collect_tokens(|this| this.parse_path(PathStyle::Type))?)
            }
            NonterminalKind::Meta => {
                token::NtMeta(P(self.collect_tokens(|this| this.parse_attr_item(false))?))
            }
            NonterminalKind::TT => token::NtTT(self.parse_token_tree()),
            NonterminalKind::Vis => token::NtVis(
                self.collect_tokens(|this| this.parse_visibility(FollowedByType::Yes))?,
            ),
            NonterminalKind::Lifetime => {
                if self.check_lifetime() {
                    token::NtLifetime(self.expect_lifetime().ident)
                } else {
                    let token_str = pprust::token_to_string(&self.token);
                    let msg = &format!("expected a lifetime, found `{}`", &token_str);
                    return Err(self.struct_span_err(self.token.span, msg));
                }
            }
        })
    }
}

/// The token is an identifier, but not `_`.
/// We prohibit passing `_` to macros expecting `ident` for now.
fn get_macro_ident(token: &Token) -> Option<(Ident, bool)> {
    token.ident().filter(|(ident, _)| ident.name != kw::Underscore)
}
