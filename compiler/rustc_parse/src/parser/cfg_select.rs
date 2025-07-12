use rustc_ast::token::Token;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::{MetaItemInner, token};
use rustc_errors::PResult;
use rustc_span::Span;

use crate::exp;
use crate::parser::Parser;

pub enum CfgSelectRule {
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
    pub unreachable: Vec<(CfgSelectRule, TokenStream, Span)>,
}

/// Parses a `TokenTree` that must be of the form `{ /* ... */ }`, and returns a `TokenStream` where
/// the surrounding braces are stripped.
fn parse_token_tree<'a>(p: &mut Parser<'a>) -> PResult<'a, TokenStream> {
    // Generate an error if the `=>` is not followed by `{`.
    if p.token != token::OpenBrace {
        p.expect(exp!(OpenBrace))?;
    }

    // Strip the outer '{' and '}'.
    match p.parse_token_tree() {
        TokenTree::Token(..) => unreachable!("because of the expect above"),
        TokenTree::Delimited(.., tts) => Ok(tts),
    }
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
                    branches.unreachable.push((CfgSelectRule::Wildcard(underscore), tts, span))
                }
            }
        } else {
            let meta_item = p.parse_meta_item_inner()?;
            p.expect(exp!(FatArrow))?;

            let tts = parse_token_tree(p)?;
            let span = meta_item.span().to(p.token.span);

            match branches.wildcard {
                None => branches.reachable.push((meta_item, tts, span)),
                Some(_) => branches.unreachable.push((CfgSelectRule::Cfg(meta_item), tts, span)),
            }
        }
    }

    Ok(branches)
}
