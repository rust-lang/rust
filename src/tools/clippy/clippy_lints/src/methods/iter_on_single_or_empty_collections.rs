use std::iter::once;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::{get_expr_use_or_unification_node, is_res_lang_ctor, path_res, std_or_core, sym};

use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::HirId;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_span::Symbol;

use super::{ITER_ON_EMPTY_COLLECTIONS, ITER_ON_SINGLE_ITEMS};

enum IterType {
    Iter,
    IterMut,
    IntoIter,
}

impl IterType {
    fn ref_prefix(&self) -> &'static str {
        match self {
            Self::Iter => "&",
            Self::IterMut => "&mut ",
            Self::IntoIter => "",
        }
    }
}

fn is_arg_ty_unified_in_fn<'tcx>(
    cx: &LateContext<'tcx>,
    fn_id: DefId,
    arg_id: HirId,
    args: impl IntoIterator<Item = &'tcx Expr<'tcx>>,
) -> bool {
    let fn_sig = cx.tcx.fn_sig(fn_id).instantiate_identity();
    let arg_id_in_args = args.into_iter().position(|e| e.hir_id == arg_id).unwrap();
    let arg_ty_in_args = fn_sig.input(arg_id_in_args).skip_binder();

    cx.tcx.predicates_of(fn_id).predicates.iter().any(|(clause, _)| {
        clause
            .as_projection_clause()
            .and_then(|p| p.map_bound(|p| p.term.as_type()).transpose())
            .is_some_and(|ty| ty.skip_binder() == arg_ty_in_args)
    }) || fn_sig
        .inputs()
        .iter()
        .enumerate()
        .any(|(i, ty)| i != arg_id_in_args && ty.skip_binder().walk().any(|arg| arg.as_type() == Some(arg_ty_in_args)))
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, method_name: Symbol, recv: &'tcx Expr<'tcx>) {
    let item = match recv.kind {
        ExprKind::Array([]) => None,
        ExprKind::Array([e]) => Some(e),
        ExprKind::Path(ref p) if is_res_lang_ctor(cx, cx.qpath_res(p, recv.hir_id), OptionNone) => None,
        ExprKind::Call(f, [arg]) if is_res_lang_ctor(cx, path_res(cx, f), OptionSome) => Some(arg),
        _ => return,
    };
    let iter_type = match method_name {
        sym::iter => IterType::Iter,
        sym::iter_mut => IterType::IterMut,
        sym::into_iter => IterType::IntoIter,
        _ => return,
    };

    let is_unified = match get_expr_use_or_unification_node(cx.tcx, expr) {
        Some((Node::Expr(parent), child_id)) => match parent.kind {
            ExprKind::If(e, _, _) | ExprKind::Match(e, _, _) if e.hir_id == child_id => false,
            ExprKind::Call(
                Expr {
                    kind: ExprKind::Path(path),
                    hir_id,
                    ..
                },
                args,
            ) => cx
                .typeck_results()
                .qpath_res(path, *hir_id)
                .opt_def_id()
                .filter(|fn_id| cx.tcx.def_kind(fn_id).is_fn_like())
                .is_some_and(|fn_id| is_arg_ty_unified_in_fn(cx, fn_id, child_id, args)),
            ExprKind::MethodCall(_name, recv, args, _span) => is_arg_ty_unified_in_fn(
                cx,
                cx.typeck_results().type_dependent_def_id(parent.hir_id).unwrap(),
                child_id,
                once(recv).chain(args.iter()),
            ),
            ExprKind::If(_, _, _)
            | ExprKind::Match(_, _, _)
            | ExprKind::Closure(_)
            | ExprKind::Ret(_)
            | ExprKind::Break(_, _) => true,
            _ => false,
        },
        Some((Node::Stmt(_) | Node::LetStmt(_), _)) => false,
        _ => true,
    };

    if is_unified {
        return;
    }

    let Some(top_crate) = std_or_core(cx) else { return };
    if let Some(i) = item {
        let sugg = format!(
            "{top_crate}::iter::once({}{})",
            iter_type.ref_prefix(),
            snippet(cx, i.span, "...")
        );
        span_lint_and_sugg(
            cx,
            ITER_ON_SINGLE_ITEMS,
            expr.span,
            format!("`{method_name}` call on a collection with only one item"),
            "try",
            sugg,
            Applicability::MaybeIncorrect,
        );
    } else {
        span_lint_and_sugg(
            cx,
            ITER_ON_EMPTY_COLLECTIONS,
            expr.span,
            format!("`{method_name}` call on an empty collection"),
            "try",
            format!("{top_crate}::iter::empty()"),
            Applicability::MaybeIncorrect,
        );
    }
}
