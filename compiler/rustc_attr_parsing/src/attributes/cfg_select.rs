use rustc_ast::token::Token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrStyle, NodeId, token};
use rustc_data_structures::fx::FxHashMap;
use rustc_feature::{AttributeTemplate, Features};
use rustc_hir::attrs::CfgEntry;
use rustc_hir::{AttrPath, Target};
use rustc_parse::exp;
use rustc_parse::parser::{Parser, Recovery};
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::UNREACHABLE_CFG_SELECT_PREDICATES;
use rustc_span::{ErrorGuaranteed, Span, Symbol, sym};

use crate::parser::MetaItemOrLitParser;
use crate::{AttributeParser, ParsedDescription, ShouldEmit, parse_cfg_entry};

#[derive(Clone)]
pub enum CfgSelectPredicate {
    Cfg(CfgEntry),
    Wildcard(Token),
}

impl CfgSelectPredicate {
    fn span(&self) -> Span {
        match self {
            CfgSelectPredicate::Cfg(cfg_entry) => cfg_entry.span(),
            CfgSelectPredicate::Wildcard(token) => token.span,
        }
    }
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
            let meta = MetaItemOrLitParser::parse_single(
                p,
                ShouldEmit::ErrorsAndLints { recovery: Recovery::Allowed },
            )
            .map_err(|diag| diag.emit())?;
            let cfg_span = meta.span();
            let cfg = AttributeParser::parse_single_args(
                sess,
                cfg_span,
                cfg_span,
                AttrStyle::Inner,
                AttrPath { segments: vec![sym::cfg_select].into_boxed_slice(), span: cfg_span },
                None,
                ParsedDescription::Macro,
                cfg_span,
                lint_node_id,
                // Doesn't matter what the target actually is here.
                Target::Crate,
                features,
                ShouldEmit::ErrorsAndLints { recovery: Recovery::Allowed },
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

    let it = branches
        .reachable
        .iter()
        .map(|(entry, _, _)| CfgSelectPredicate::Cfg(entry.clone()))
        .chain(branches.wildcard.as_ref().map(|(t, _, _)| CfgSelectPredicate::Wildcard(*t)))
        .chain(branches.unreachable.iter().map(|(entry, _, _)| CfgSelectPredicate::clone(entry)));

    lint_unreachable(p, it, lint_node_id);

    Ok(branches)
}

fn lint_unreachable(
    p: &mut Parser<'_>,
    predicates: impl Iterator<Item = CfgSelectPredicate>,
    lint_node_id: NodeId,
) {
    // Symbols that have a known value.
    let mut known = FxHashMap::<Symbol, bool>::default();
    let mut wildcard_span = None;
    let mut it = predicates;

    let branch_is_unreachable = |predicate: CfgSelectPredicate, wildcard_span| {
        let span = predicate.span();
        p.psess.buffer_lint(
            UNREACHABLE_CFG_SELECT_PREDICATES,
            span,
            lint_node_id,
            BuiltinLintDiag::UnreachableCfg { span, wildcard_span },
        );
    };

    for predicate in &mut it {
        let CfgSelectPredicate::Cfg(ref cfg_entry) = predicate else {
            wildcard_span = Some(predicate.span());
            break;
        };

        match cfg_entry {
            CfgEntry::Bool(true, _) => {
                wildcard_span = Some(predicate.span());
                break;
            }
            CfgEntry::Bool(false, _) => continue,
            CfgEntry::NameValue { name, value, .. } => match value {
                None => {
                    // `name` will be false in all subsequent branches.
                    let current = known.insert(*name, false);

                    match current {
                        None => continue,
                        Some(false) => {
                            branch_is_unreachable(predicate, None);
                            break;
                        }
                        Some(true) => {
                            // this branch will be taken, so all subsequent branches are unreachable.
                            break;
                        }
                    }
                }
                Some(_) => { /* for now we don't bother solving these */ }
            },
            CfgEntry::Not(inner, _) => match &**inner {
                CfgEntry::NameValue { name, value: None, .. } => {
                    // `name` will be true in all subsequent branches.
                    let current = known.insert(*name, true);

                    match current {
                        None => continue,
                        Some(true) => {
                            branch_is_unreachable(predicate, None);
                            break;
                        }
                        Some(false) => {
                            // this branch will be taken, so all subsequent branches are unreachable.
                            break;
                        }
                    }
                }
                _ => { /* for now we don't bother solving these */ }
            },
            CfgEntry::All(_, _) | CfgEntry::Any(_, _) => {
                /* for now we don't bother solving these */
            }
            CfgEntry::Version(..) => { /* don't bother solving these */ }
        }
    }

    for predicate in it {
        branch_is_unreachable(predicate, wildcard_span)
    }
}
