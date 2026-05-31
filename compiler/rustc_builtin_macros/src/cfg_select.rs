use rustc_ast::attr::{AttrIdGenerator, mk_attr_from_item};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    AttrItem, AttrItemKind, EarlyParsedAttribute, Expr, Path, Safety, ast, token,
    tokenstream as tts,
};
use rustc_attr_parsing as attr;
use rustc_attr_parsing::{CfgSelectBranches, EvalConfigResult, parse_cfg_select};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacResult, MacroExpanderResult};
use rustc_expand::expand::DeclaredIdents;
use rustc_hir::attrs::CfgEntry;
use rustc_span::{Ident, Span, sym};
use smallvec::SmallVec;

use crate::errors::CfgSelectNoMatches;

/// This intermediate structure is used to emit parse errors for the branches that are not chosen.
/// The `MacResult` instance below parses all branches, emitting any errors it encounters, but only
/// keeps the parse result for the selected branch.
struct CfgSelectResult<'cx, 'sess> {
    ecx: &'cx mut ExtCtxt<'sess>,
    site_span: Span,
    selected_tts: TokenStream,
    selected_span: Span,
    other_branches: CfgSelectBranches,
    cfg_entry: CfgEntry,
}

fn tts_to_mac_result<'cx, 'sess>(
    ecx: &'cx mut ExtCtxt<'sess>,
    site_span: Span,
    tts: TokenStream,
    span: Span,
) -> Box<dyn MacResult + 'cx> {
    match ExpandResult::from_tts(ecx, tts, site_span, span, Ident::with_dummy_span(sym::cfg_select))
    {
        ExpandResult::Ready(x) => x,
        _ => unreachable!("from_tts always returns Ready"),
    }
}

macro_rules! forward_to_parser_any_macro {
    ($method_name:ident, $ret_ty:ty) => {
        fn $method_name(self: Box<Self>) -> Option<$ret_ty> {
            let CfgSelectResult { ecx, site_span, selected_tts, selected_span, .. } = *self;

            for (_, tts, span) in self.other_branches.into_iter_tts() {
                let _ = tts_to_mac_result(ecx, site_span, tts, span).$method_name();
            }

            tts_to_mac_result(ecx, site_span, selected_tts, selected_span).$method_name()
        }
    };

    (make_items) => {
        // The same logic as above, but we also register the items that were not selected in the
        // resolver for error reporting, as well as annotate the selected item with `#[cfg_trace]`.
        fn make_items(self: Box<Self>) -> Option<SmallVec<[Box<ast::Item>; 1]>> {
            let CfgSelectResult { ecx, site_span, selected_tts, selected_span, cfg_entry, .. } =
                *self;

            for (cfg_entry, tts, span) in self.other_branches.into_iter_tts() {
                if let Some(items) = tts_to_mac_result(ecx, site_span, tts, span).make_items() {
                    // Register item names that were not selected for error reporting. We do this
                    // for `#[cfg]` too.
                    for item in items {
                        for name in item.declared_idents() {
                            ecx.resolver.append_stripped_cfg_item(
                                ecx.current_expansion.lint_node_id,
                                name,
                                cfg_entry.clone(),
                                span,
                            );
                        }
                    }
                }
            }

            tts_to_mac_result(ecx, site_span, selected_tts, selected_span).make_items().map(
                |items| {
                    items
                        .into_iter()
                        .map(|mut item| {
                            item.attrs.push(mk_attr(
                                &ecx.sess.psess.attr_id_generator,
                                cfg_entry.clone(),
                            ));
                            item
                        })
                        .collect()
                },
            )
        }
    };
}

/// Construct a `#[<cfg_trace>]` attribute from a `CfgEntry`. This allows us to keep track of items
/// that were behind a `cfg_select!`, which is relevant for some diagnostics.
fn mk_attr(g: &AttrIdGenerator, cfg_entry: CfgEntry) -> ast::Attribute {
    let cfg_span = cfg_entry.span();
    let args = AttrItemKind::Parsed(EarlyParsedAttribute::CfgTrace(cfg_entry));
    let trees = vec![
        tts::AttrTokenTree::Token(
            token::Token { kind: token::TokenKind::Pound, span: cfg_span },
            tts::Spacing::JointHidden,
        ),
        tts::AttrTokenTree::Delimited(
            tts::DelimSpan::dummy(),
            tts::DelimSpacing::new(tts::Spacing::JointHidden, tts::Spacing::Alone),
            token::Delimiter::Bracket,
            tts::AttrTokenStream::new(vec![tts::AttrTokenTree::Token(
                token::Token {
                    kind: token::TokenKind::Ident(sym::cfg_trace, token::IdentIsRaw::No),
                    span: cfg_span,
                },
                tts::Spacing::Alone,
            )]),
        ),
    ];
    let tokens = Some(tts::LazyAttrTokenStream::new_direct(tts::AttrTokenStream::new(trees)));
    let attr_item = AttrItem {
        unsafety: Safety::Default,
        path: Path::from_ident(Ident::new(sym::cfg_trace, cfg_span)),
        args,
        tokens: None,
    };
    mk_attr_from_item(g, attr_item, tokens, ast::AttrStyle::Outer, cfg_span)
}

impl<'cx, 'sess> MacResult for CfgSelectResult<'cx, 'sess> {
    forward_to_parser_any_macro!(make_expr, Box<Expr>);
    forward_to_parser_any_macro!(make_stmts, SmallVec<[ast::Stmt; 1]>);
    forward_to_parser_any_macro!(make_items);

    forward_to_parser_any_macro!(make_impl_items, SmallVec<[Box<ast::AssocItem>; 1]>);
    forward_to_parser_any_macro!(make_trait_impl_items, SmallVec<[Box<ast::AssocItem>; 1]>);
    forward_to_parser_any_macro!(make_trait_items, SmallVec<[Box<ast::AssocItem>; 1]>);
    forward_to_parser_any_macro!(make_foreign_items, SmallVec<[Box<ast::ForeignItem>; 1]>);

    forward_to_parser_any_macro!(make_ty, Box<ast::Ty>);
    forward_to_parser_any_macro!(make_pat, Box<ast::Pat>);
}

pub(super) fn expand_cfg_select<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    ExpandResult::Ready(
        match parse_cfg_select(
            &mut ecx.new_parser_from_tts(tts),
            ecx.sess,
            Some(ecx.ecfg.features),
            ecx.current_expansion.lint_node_id,
        ) {
            Ok(mut branches) => {
                if let Some((cfg_entry, selected_tts, selected_span)) =
                    branches.pop_first_match(|cfg| {
                        matches!(attr::eval_config_entry(&ecx.sess, cfg), EvalConfigResult::True)
                    })
                {
                    let mac = CfgSelectResult {
                        ecx,
                        selected_tts,
                        selected_span,
                        other_branches: branches,
                        site_span: sp,
                        cfg_entry,
                    };
                    return ExpandResult::Ready(Box::new(mac));
                } else {
                    // Emit a compiler error when none of the predicates matched.
                    let guar = ecx.dcx().emit_err(CfgSelectNoMatches { span: sp });
                    DummyResult::any(sp, guar)
                }
            }
            Err(guar) => DummyResult::any(sp, guar),
        },
    )
}
