use rustc_ast::token::Token;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::{MetaItemInner, token};
use rustc_errors::PResult;
use rustc_span::Span;

use crate::exp;
use crate::parser::{AttrWrapper, ForceCollect, Parser, Restrictions, Trailing, UsePreAttrPos};

pub enum CfgSelectPredicate {
    Cfg(MetaItemInner),
    Wildcard(Token),
}

#[derive(Default)]
pub struct CfgSelectBranches {
    /// All the conditional branches.
    pub reachable: Vec<(MetaItemInner, TokenStream, Span)>,
    /// The first wildcard `_ => { ... }` branch.
    pub wildcard: Option<(Token, TokenStream, Span)>,
    /// All branches after the first wildcard, including further wildcards.
    /// These branches are kept for formatting.
    pub unreachable: Vec<(CfgSelectPredicate, TokenStream, Span)>,
}

/// Parses a `TokenTree` consisting either of `{ /* ... */ }` (and strip the braces) or an
/// expression followed by a comma (and strip the comma).
fn parse_token_tree<'a>(p: &mut Parser<'a>) -> PResult<'a, TokenStream> {
    if p.token == token::OpenBrace {
        // Strip the outer '{' and '}'.
        match p.parse_token_tree() {
            TokenTree::Token(..) => unreachable!("because of the expect above"),
            TokenTree::Delimited(.., tts) => return Ok(tts),
        }
    }
    let expr = p.collect_tokens(None, AttrWrapper::empty(), ForceCollect::Yes, |p, _| {
        p.parse_expr_res(Restrictions::STMT_EXPR, AttrWrapper::empty())
            .map(|(expr, _)| (expr, Trailing::No, UsePreAttrPos::No))
    })?;
    if !classify::expr_is_complete(&expr) && p.token != token::CloseBrace && p.token != token::Eof {
        p.expect(exp!(Comma))?;
    } else {
        let _ = p.eat(exp!(Comma));
    }
    Ok(TokenStream::from_ast(&expr))
}

pub fn parse_cfg_select<'a>(p: &mut Parser<'a>) -> PResult<'a, CfgSelectBranches> {
    let mut branches = CfgSelectBranches::default();

    while p.token != token::Eof {
        if p.eat_keyword(exp!(Underscore)) {
            let underscore = p.prev_token;
            p.expect(exp!(FatArrow))?;

            let tts = parse_token_tree(p)?;
            let span = underscore.span.to(p.token.span);

            match branches.wildcard {
                None => branches.wildcard = Some((underscore, tts, span)),
                Some(_) => {
                    branches.unreachable.push((CfgSelectPredicate::Wildcard(underscore), tts, span))
                }
            }
        } else {
            let meta_item = p.parse_meta_item_inner()?;
            p.expect(exp!(FatArrow))?;

            let tts = parse_token_tree(p)?;
            let span = meta_item.span().to(p.token.span);

            match branches.wildcard {
                None => branches.reachable.push((meta_item, tts, span)),
                Some(_) => {
                    branches.unreachable.push((CfgSelectPredicate::Cfg(meta_item), tts, span))
                }
            }
        }
    }

    Ok(branches)
}
