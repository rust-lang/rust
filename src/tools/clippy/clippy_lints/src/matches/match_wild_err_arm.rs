use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::macros::{is_panic, root_macro_call};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{in_constant, is_wild, peel_blocks_with_stmt};
use rustc_hir::{Arm, Expr, PatKind};
use rustc_lint::LateContext;
use rustc_span::symbol::{kw, sym};

use super::MATCH_WILD_ERR_ARM;

pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, ex: &Expr<'tcx>, arms: &[Arm<'tcx>]) {
    // `unwrap`/`expect` is not (yet) const, so we want to allow this in const contexts for now
    if in_constant(cx, ex.hir_id) {
        return;
    }

    let ex_ty = cx.typeck_results().expr_ty(ex).peel_refs();
    if is_type_diagnostic_item(cx, ex_ty, sym::Result) {
        for arm in arms {
            if let PatKind::TupleStruct(ref path, inner, _) = arm.pat.kind {
                let path_str = rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false));
                if path_str == "Err" {
                    let mut matching_wild = inner.iter().any(is_wild);
                    let mut ident_bind_name = kw::Underscore;
                    if !matching_wild {
                        // Looking for unused bindings (i.e.: `_e`)
                        for pat in inner {
                            if let PatKind::Binding(_, id, ident, None) = pat.kind {
                                if ident.as_str().starts_with('_') && !is_local_used(cx, arm.body, id) {
                                    ident_bind_name = ident.name;
                                    matching_wild = true;
                                }
                            }
                        }
                    }
                    if_chain! {
                        if matching_wild;
                        if let Some(macro_call) = root_macro_call(peel_blocks_with_stmt(arm.body).span);
                        if is_panic(cx, macro_call.def_id);
                        then {
                            // `Err(_)` or `Err(_e)` arm with `panic!` found
                            span_lint_and_note(cx,
                                MATCH_WILD_ERR_ARM,
                                arm.pat.span,
                                &format!("`Err({ident_bind_name})` matches all errors"),
                                None,
                                "match each error separately or use the error output, or use `.expect(msg)` if the error case is unreachable",
                            );
                        }
                    }
                }
            }
        }
    }
}
