use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::{get_expr_use_or_unification_node, is_lang_ctor, is_no_std_crate};

use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;

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

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, method_name: &str, recv: &Expr<'_>) {
    let item = match &recv.kind {
        ExprKind::Array(v) if v.len() <= 1 => v.first(),
        ExprKind::Path(p) => {
            if is_lang_ctor(cx, p, OptionNone) {
                None
            } else {
                return;
            }
        },
        ExprKind::Call(f, some_args) if some_args.len() == 1 => {
            if let ExprKind::Path(p) = &f.kind {
                if is_lang_ctor(cx, p, OptionSome) {
                    Some(&some_args[0])
                } else {
                    return;
                }
            } else {
                return;
            }
        },
        _ => return,
    };
    let iter_type = match method_name {
        "iter" => IterType::Iter,
        "iter_mut" => IterType::IterMut,
        "into_iter" => IterType::IntoIter,
        _ => return,
    };

    let is_unified = match get_expr_use_or_unification_node(cx.tcx, expr) {
        Some((Node::Expr(parent), child_id)) => match parent.kind {
            ExprKind::If(e, _, _) | ExprKind::Match(e, _, _) if e.hir_id == child_id => false,
            ExprKind::If(_, _, _)
            | ExprKind::Match(_, _, _)
            | ExprKind::Closure(_)
            | ExprKind::Ret(_)
            | ExprKind::Break(_, _) => true,
            _ => false,
        },
        Some((Node::Stmt(_) | Node::Local(_), _)) => false,
        _ => true,
    };

    if is_unified {
        return;
    }

    if let Some(i) = item {
        let sugg = format!(
            "{}::iter::once({}{})",
            if is_no_std_crate(cx) { "core" } else { "std" },
            iter_type.ref_prefix(),
            snippet(cx, i.span, "...")
        );
        span_lint_and_sugg(
            cx,
            ITER_ON_SINGLE_ITEMS,
            expr.span,
            &format!("`{method_name}` call on a collection with only one item"),
            "try",
            sugg,
            Applicability::MaybeIncorrect,
        );
    } else {
        span_lint_and_sugg(
            cx,
            ITER_ON_EMPTY_COLLECTIONS,
            expr.span,
            &format!("`{method_name}` call on an empty collection"),
            "try",
            if is_no_std_crate(cx) {
                "core::iter::empty()".to_string()
            } else {
                "std::iter::empty()".to_string()
            },
            Applicability::MaybeIncorrect,
        );
    }
}
