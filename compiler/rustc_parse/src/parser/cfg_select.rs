use rustc_ast::token;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_errors::PResult;
use rustc_span::Span;

use crate::exp;
use crate::parser::{AttrWrapper, ForceCollect, Parser, Restrictions, Trailing, UsePreAttrPos};

impl<'a> Parser<'a> {
    /// Parses a `TokenTree` consisting either of `{ /* ... */ }` optionally followed by a comma
    /// (and strip the braces and the optional comma) or an expression followed by a comma
    /// (and strip the comma).
    pub fn parse_delimited_token_tree(&mut self) -> PResult<'a, TokenStream> {
        if self.token == token::OpenBrace {
            // Strip the outer '{' and '}'.
            match self.parse_token_tree() {
                TokenTree::Token(..) => unreachable!("because the current token is a '{{'"),
                TokenTree::Delimited(.., tts) => {
                    // Optionally end with a comma.
                    let _ = self.eat(exp!(Comma));
                    return Ok(tts);
                }
            }
        }
        let expr = self.collect_tokens(None, AttrWrapper::empty(), ForceCollect::Yes, |p, _| {
            p.parse_expr_res(Restrictions::STMT_EXPR, AttrWrapper::empty())
                .map(|(expr, _)| (expr, Trailing::No, UsePreAttrPos::No))
        })?;
        if !classify::expr_is_complete(&expr)
            && self.token != token::CloseBrace
            && self.token != token::Eof
        {
            self.expect(exp!(Comma))?;
        } else {
            let _ = self.eat(exp!(Comma));
        }
        Ok(TokenStream::from_ast(&expr))
    }

    /// Parses outer attributes before a `cfg_select!` branch for recovery.
    pub fn parse_cfg_select_branch_outer_attrs(&mut self) -> PResult<'a, Option<Vec<Span>>> {
        let attrs = self.parse_outer_attributes()?;
        if attrs.is_empty() {
            return Ok(None);
        }

        Ok(Some(attrs.take_for_recovery(self.psess).into_iter().map(|attr| attr.span).collect()))
    }
}
