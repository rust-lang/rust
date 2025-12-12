use rustc_ast::token::Token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrStyle, NodeId, token};
use rustc_feature::{AttributeTemplate, Features};
use rustc_hir::AttrPath;
use rustc_hir::attrs::CfgEntry;
use rustc_parse::exp;
use rustc_parse::parser::Parser;
use rustc_session::Session;
use rustc_span::{ErrorGuaranteed, Ident, Span};

use crate::parser::MetaItemOrLitParser;
use crate::{AttributeParser, ParsedDescription, ShouldEmit, parse_cfg_entry};

pub enum CfgSelectPredicate {
    Cfg(CfgEntry),
    Wildcard(Token),
}

#[derive(Default)]
pub struct CfgSelectBranches {
    /// All the conditional branches.
    pub reachable: Vec<(CfgEntry, TokenStream, Span)>,
    /// The first wildcard `_ => { ... }` branch.
    pub wildcard: Option<(Token, TokenStream, Span)>,
    /// All branches after the first wildcard, including further wildcards.
    /// These branches are kept for formatting.
    pub unreachable: Vec<(CfgSelectPredicate, TokenStream, Span)>,
}

impl CfgSelectBranches {
    /// Removes the top-most branch for which `predicate` returns `true`,
    /// or the wildcard if none of the reachable branches satisfied the predicate.
    pub fn pop_first_match<F>(&mut self, predicate: F) -> Option<(TokenStream, Span)>
    where
        F: Fn(&CfgEntry) -> bool,
    {
        for (index, (cfg, _, _)) in self.reachable.iter().enumerate() {
            if predicate(cfg) {
                let matched = self.reachable.remove(index);
                return Some((matched.1, matched.2));
            }
        }

        self.wildcard.take().map(|(_, tts, span)| (tts, span))
    }

    /// Consume this value and iterate over all the `TokenStream`s that it stores.
    pub fn into_iter_tts(self) -> impl Iterator<Item = (TokenStream, Span)> {
        let it1 = self.reachable.into_iter().map(|(_, tts, span)| (tts, span));
        let it2 = self.wildcard.into_iter().map(|(_, tts, span)| (tts, span));
        let it3 = self.unreachable.into_iter().map(|(_, tts, span)| (tts, span));

        it1.chain(it2).chain(it3)
    }
}

pub fn parse_cfg_select(
    p: &mut Parser<'_>,
    sess: &Session,
    features: Option<&Features>,
    lint_node_id: NodeId,
) -> Result<CfgSelectBranches, ErrorGuaranteed> {
    let mut branches = CfgSelectBranches::default();

    while p.token != token::Eof {
        if p.eat_keyword(exp!(Underscore)) {
            let underscore = p.prev_token;
            p.expect(exp!(FatArrow)).map_err(|e| e.emit())?;

            let tts = p.parse_delimited_token_tree().map_err(|e| e.emit())?;
            let span = underscore.span.to(p.token.span);

            match branches.wildcard {
                None => branches.wildcard = Some((underscore, tts, span)),
                Some(_) => {
                    branches.unreachable.push((CfgSelectPredicate::Wildcard(underscore), tts, span))
                }
            }
        } else {
            let meta = MetaItemOrLitParser::parse_single(p, ShouldEmit::ErrorsAndLints)
                .map_err(|diag| diag.emit())?;
            let cfg_span = meta.span();
            let cfg = AttributeParser::parse_single_args(
                sess,
                cfg_span,
                cfg_span,
                AttrStyle::Inner,
                AttrPath {
                    segments: vec![Ident::from_str("cfg_select")].into_boxed_slice(),
                    span: cfg_span,
                },
                None,
                ParsedDescription::Macro,
                cfg_span,
                lint_node_id,
                features,
                ShouldEmit::ErrorsAndLints,
                &meta,
                parse_cfg_entry,
                &AttributeTemplate::default(),
            )?;

            p.expect(exp!(FatArrow)).map_err(|e| e.emit())?;

            let tts = p.parse_delimited_token_tree().map_err(|e| e.emit())?;
            let span = cfg_span.to(p.token.span);

            match branches.wildcard {
                None => branches.reachable.push((cfg, tts, span)),
                Some(_) => branches.unreachable.push((CfgSelectPredicate::Cfg(cfg), tts, span)),
            }
        }
    }

    Ok(branches)
}
