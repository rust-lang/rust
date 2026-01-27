use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{Expr, ast};
use rustc_attr_parsing as attr;
use rustc_attr_parsing::{
    CfgSelectBranches, CfgSelectPredicate, EvalConfigResult, parse_cfg_select,
};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacResult, MacroExpanderResult};
use rustc_span::{Ident, Span, sym};
use smallvec::SmallVec;

use crate::errors::{CfgSelectNoMatches, CfgSelectUnreachable};

/// This intermediate structure is used to emit parse errors for the branches that are not chosen.
/// The `MacResult` instance below parses all branches, emitting any errors it encounters, but only
/// keeps the parse result for the selected branch.
struct CfgSelectResult<'cx, 'sess> {
    ecx: &'cx mut ExtCtxt<'sess>,
    site_span: Span,
    selected_tts: TokenStream,
    selected_span: Span,
    other_branches: CfgSelectBranches,
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

            for (tts, span) in self.other_branches.into_iter_tts() {
                let _ = tts_to_mac_result(ecx, site_span, tts, span).$method_name();
            }

            tts_to_mac_result(ecx, site_span, selected_tts, selected_span).$method_name()
        }
    };
}

impl<'cx, 'sess> MacResult for CfgSelectResult<'cx, 'sess> {
    forward_to_parser_any_macro!(make_expr, Box<Expr>);
    forward_to_parser_any_macro!(make_stmts, SmallVec<[ast::Stmt; 1]>);
    forward_to_parser_any_macro!(make_items, SmallVec<[Box<ast::Item>; 1]>);

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
                if let Some((underscore, _, _)) = branches.wildcard {
                    // Warn for every unreachable predicate. We store the fully parsed branch for rustfmt.
                    for (predicate, _, _) in &branches.unreachable {
                        let span = match predicate {
                            CfgSelectPredicate::Wildcard(underscore) => underscore.span,
                            CfgSelectPredicate::Cfg(cfg) => cfg.span(),
                        };
                        let err = CfgSelectUnreachable { span, wildcard_span: underscore.span };
                        ecx.dcx().emit_warn(err);
                    }
                }

                if let Some((selected_tts, selected_span)) = branches.pop_first_match(|cfg| {
                    matches!(attr::eval_config_entry(&ecx.sess, cfg), EvalConfigResult::True)
                }) {
                    let mac = CfgSelectResult {
                        ecx,
                        selected_tts,
                        selected_span,
                        other_branches: branches,
                        site_span: sp,
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
