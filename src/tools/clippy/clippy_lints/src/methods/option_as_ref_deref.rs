use crate::utils::{
    is_type_diagnostic_item, match_def_path, meets_msrv, path_to_local_id, paths, remove_blocks, snippet,
    span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_semver::RustcVersion;
use rustc_span::sym;

use super::OPTION_AS_REF_DEREF;

const OPTION_AS_REF_DEREF_MSRV: RustcVersion = RustcVersion::new(1, 40, 0);

/// lint use of `_.as_ref().map(Deref::deref)` for `Option`s
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    as_ref_args: &[hir::Expr<'_>],
    map_args: &[hir::Expr<'_>],
    is_mut: bool,
    msrv: Option<&RustcVersion>,
) {
    if !meets_msrv(msrv, &OPTION_AS_REF_DEREF_MSRV) {
        return;
    }

    let same_mutability = |m| (is_mut && m == &hir::Mutability::Mut) || (!is_mut && m == &hir::Mutability::Not);

    let option_ty = cx.typeck_results().expr_ty(&as_ref_args[0]);
    if !is_type_diagnostic_item(cx, option_ty, sym::option_type) {
        return;
    }

    let deref_aliases: [&[&str]; 9] = [
        &paths::DEREF_TRAIT_METHOD,
        &paths::DEREF_MUT_TRAIT_METHOD,
        &paths::CSTRING_AS_C_STR,
        &paths::OS_STRING_AS_OS_STR,
        &paths::PATH_BUF_AS_PATH,
        &paths::STRING_AS_STR,
        &paths::STRING_AS_MUT_STR,
        &paths::VEC_AS_SLICE,
        &paths::VEC_AS_MUT_SLICE,
    ];

    let is_deref = match map_args[1].kind {
        hir::ExprKind::Path(ref expr_qpath) => cx
            .qpath_res(expr_qpath, map_args[1].hir_id)
            .opt_def_id()
            .map_or(false, |fun_def_id| {
                deref_aliases.iter().any(|path| match_def_path(cx, fun_def_id, path))
            }),
        hir::ExprKind::Closure(_, _, body_id, _, _) => {
            let closure_body = cx.tcx.hir().body(body_id);
            let closure_expr = remove_blocks(&closure_body.value);

            match &closure_expr.kind {
                hir::ExprKind::MethodCall(_, _, args, _) => {
                    if_chain! {
                        if args.len() == 1;
                        if path_to_local_id(&args[0], closure_body.params[0].pat.hir_id);
                        let adj = cx
                            .typeck_results()
                            .expr_adjustments(&args[0])
                            .iter()
                            .map(|x| &x.kind)
                            .collect::<Box<[_]>>();
                        if let [ty::adjustment::Adjust::Deref(None), ty::adjustment::Adjust::Borrow(_)] = *adj;
                        then {
                            let method_did = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id).unwrap();
                            deref_aliases.iter().any(|path| match_def_path(cx, method_did, path))
                        } else {
                            false
                        }
                    }
                },
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, m, ref inner) if same_mutability(m) => {
                    if_chain! {
                        if let hir::ExprKind::Unary(hir::UnOp::Deref, ref inner1) = inner.kind;
                        if let hir::ExprKind::Unary(hir::UnOp::Deref, ref inner2) = inner1.kind;
                        then {
                            path_to_local_id(inner2, closure_body.params[0].pat.hir_id)
                        } else {
                            false
                        }
                    }
                },
                _ => false,
            }
        },
        _ => false,
    };

    if is_deref {
        let current_method = if is_mut {
            format!(".as_mut().map({})", snippet(cx, map_args[1].span, ".."))
        } else {
            format!(".as_ref().map({})", snippet(cx, map_args[1].span, ".."))
        };
        let method_hint = if is_mut { "as_deref_mut" } else { "as_deref" };
        let hint = format!("{}.{}()", snippet(cx, as_ref_args[0].span, ".."), method_hint);
        let suggestion = format!("try using {} instead", method_hint);

        let msg = format!(
            "called `{0}` on an Option value. This can be done more directly \
            by calling `{1}` instead",
            current_method, hint
        );
        span_lint_and_sugg(
            cx,
            OPTION_AS_REF_DEREF,
            expr.span,
            &msg,
            &suggestion,
            hint,
            Applicability::MachineApplicable,
        );
    }
}
