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
