use rustc_ast::tokenstream::TokenStream;
use rustc_attr_parsing as attr;
use rustc_attr_parsing::{
    CfgSelectBranches, CfgSelectPredicate, EvalConfigResult, ShouldEmit, parse_cfg_select,
};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::{Ident, Span, sym};

use crate::errors::{CfgSelectNoMatches, CfgSelectUnreachable};

/// Selects the first arm whose predicate evaluates to true.
fn select_arm(ecx: &ExtCtxt<'_>, branches: CfgSelectBranches) -> Option<(TokenStream, Span)> {
    let mut result = None;
    for (cfg, tt, arm_span) in branches.reachable {
        if let EvalConfigResult::True = attr::eval_config_entry(
            &ecx.sess,
            &cfg,
            ecx.current_expansion.lint_node_id,
            ShouldEmit::ErrorsAndLints,
        ) {
            // FIXME(#149215) Ideally we should short-circuit here, but `eval_config_entry` currently emits lints so we cannot do this yet.
            result.get_or_insert((tt, arm_span));
        }
    }

    let wildcard = branches.wildcard.map(|(_, tt, span)| (tt, span));
    result.or(wildcard)
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
            Ok(branches) => {
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

                if let Some((tts, arm_span)) = select_arm(ecx, branches) {
                    return ExpandResult::from_tts(
                        ecx,
                        tts,
                        sp,
                        arm_span,
                        Ident::with_dummy_span(sym::cfg_select),
                    );
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
