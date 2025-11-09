use rustc_ast::token::Token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{MetaItemInner, token};
use rustc_errors::PResult;
use rustc_parse::exp;
use rustc_parse::parser::Parser;
use rustc_span::Span;

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

pub fn parse_cfg_select<'a>(p: &mut Parser<'a>) -> PResult<'a, CfgSelectBranches> {
    let mut branches = CfgSelectBranches::default();

    while p.token != token::Eof {
        if p.eat_keyword(exp!(Underscore)) {
            let underscore = p.prev_token;
            p.expect(exp!(FatArrow))?;

            let tts = p.parse_delimited_token_tree()?;
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

            let tts = p.parse_delimited_token_tree()?;
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
