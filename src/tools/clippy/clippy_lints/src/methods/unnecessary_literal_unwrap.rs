use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{MaybePath, is_res_lang_ctor, last_path_segment, path_res, sym};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, AmbigArg};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_span::Symbol;

use super::UNNECESSARY_LITERAL_UNWRAP;

fn get_ty_from_args<'a>(args: Option<&'a [hir::GenericArg<'a>]>, index: usize) -> Option<&'a hir::Ty<'a, AmbigArg>> {
    let args = args?;

    if args.len() <= index {
        return None;
    }

    match args[index] {
        hir::GenericArg::Type(ty) => Some(ty),
        _ => None,
    }
}

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    method: Symbol,
    args: &[hir::Expr<'_>],
) {
    let init = clippy_utils::expr_or_init(cx, recv);
    if init.span.from_expansion() {
        // don't lint if the receiver or binding initializer comes from a macro
        // (e.g. `let x = option_env!(..); x.unwrap()`)
        return;
    }

    let (constructor, call_args, ty) = if let hir::ExprKind::Call(call, call_args) = init.kind {
        let Some(qpath) = call.qpath_opt() else { return };

        let args = last_path_segment(qpath).args.map(|args| args.args);
        let res = cx.qpath_res(qpath, call.hir_id());

        if is_res_lang_ctor(cx, res, hir::LangItem::OptionSome) {
            (sym::Some, call_args, get_ty_from_args(args, 0))
        } else if is_res_lang_ctor(cx, res, hir::LangItem::ResultOk) {
            (sym::Ok, call_args, get_ty_from_args(args, 0))
        } else if is_res_lang_ctor(cx, res, hir::LangItem::ResultErr) {
            (sym::Err, call_args, get_ty_from_args(args, 1))
        } else {
            return;
        }
    } else if is_res_lang_ctor(cx, path_res(cx, init), hir::LangItem::OptionNone) {
        let call_args: &[hir::Expr<'_>] = &[];
        (sym::None, call_args, None)
    } else {
        return;
    };

    let help_message = format!("used `{method}()` on `{constructor}` value");
    let suggestion_message = format!("remove the `{constructor}` and `{method}()`");

    span_lint_and_then(cx, UNNECESSARY_LITERAL_UNWRAP, expr.span, help_message, |diag| {
        let suggestions = match (constructor, method, ty) {
            (sym::None, sym::unwrap, _) => Some(vec![(expr.span, "panic!()".to_string())]),
            (sym::None, sym::expect, _) => Some(vec![
                (expr.span.with_hi(args[0].span.lo()), "panic!(".to_string()),
                (expr.span.with_lo(args[0].span.hi()), ")".to_string()),
            ]),
            (sym::Some | sym::Ok, sym::unwrap_unchecked, _) | (sym::Err, sym::unwrap_err_unchecked, _) => {
                let mut suggs = vec![
                    (recv.span.with_hi(call_args[0].span.lo()), String::new()),
                    (expr.span.with_lo(call_args[0].span.hi()), String::new()),
                ];
                // try to also remove the unsafe block if present
                if let hir::Node::Block(block) = cx.tcx.parent_hir_node(expr.hir_id)
                    && let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = block.rules
                {
                    suggs.extend([
                        (block.span.shrink_to_lo().to(expr.span.shrink_to_lo()), String::new()),
                        (expr.span.shrink_to_hi().to(block.span.shrink_to_hi()), String::new()),
                    ]);
                }
                Some(suggs)
            },
            (sym::None, sym::unwrap_or_default, _) => {
                let ty = cx.typeck_results().expr_ty(expr);
                let default_ty_string = if let ty::Adt(def, ..) = ty.kind() {
                    with_forced_trimmed_paths!(format!("{}", cx.tcx.def_path_str(def.did())))
                } else {
                    "Default".to_string()
                };
                Some(vec![(expr.span, format!("{default_ty_string}::default()"))])
            },
            (sym::None, sym::unwrap_or, _) => Some(vec![
                (expr.span.with_hi(args[0].span.lo()), String::new()),
                (expr.span.with_lo(args[0].span.hi()), String::new()),
            ]),
            (sym::None, sym::unwrap_or_else, _) => match args[0].kind {
                hir::ExprKind::Closure(hir::Closure { body, .. }) => Some(vec![
                    (expr.span.with_hi(cx.tcx.hir_body(*body).value.span.lo()), String::new()),
                    (expr.span.with_lo(args[0].span.hi()), String::new()),
                ]),
                _ => None,
            },
            _ if call_args.is_empty() => None,
            (_, _, Some(_)) => None,
            (sym::Ok, sym::unwrap_err, None) | (sym::Err, sym::unwrap, None) => Some(vec![
                (
                    recv.span.with_hi(call_args[0].span.lo()),
                    "panic!(\"{:?}\", ".to_string(),
                ),
                (expr.span.with_lo(call_args[0].span.hi()), ")".to_string()),
            ]),
            (sym::Ok, sym::expect_err, None) | (sym::Err, sym::expect, None) => Some(vec![
                (
                    recv.span.with_hi(call_args[0].span.lo()),
                    "panic!(\"{1}: {:?}\", ".to_string(),
                ),
                (call_args[0].span.with_lo(args[0].span.lo()), ", ".to_string()),
            ]),
            (_, _, None) => Some(vec![
                (recv.span.with_hi(call_args[0].span.lo()), String::new()),
                (expr.span.with_lo(call_args[0].span.hi()), String::new()),
            ]),
        };

        match (init.span == recv.span, suggestions) {
            (true, Some(suggestions)) => {
                diag.multipart_suggestion(suggestion_message, suggestions, Applicability::MachineApplicable);
            },
            _ => {
                diag.span_help(init.span, suggestion_message);
            },
        }
    });
}
