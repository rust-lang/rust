use rustc_ast::token;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_errors::PResult;

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
}
