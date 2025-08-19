use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{is_from_proc_macro, path_to_local};
use itertools::Itertools;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use std::iter::once;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for tuple<=>array conversions that are not done with `.into()`.
    ///
    /// ### Why is this bad?
    /// It may be unnecessary complexity. `.into()` works for converting tuples<=> arrays of up to
    /// 12 elements and conveys the intent more clearly, while also leaving less room for hard to
    /// spot bugs!
    ///
    /// ### Known issues
    /// The suggested code may hide potential asymmetry in some cases. See
    /// [#11085](https://github.com/rust-lang/rust-clippy/issues/11085) for more info.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&t| t.into()).collect();
    /// ```
    #[clippy::version = "1.72.0"]
    pub TUPLE_ARRAY_CONVERSIONS,
    nursery,
    "checks for tuple<=>array conversions that are not done with `.into()`"
}
impl_lint_pass!(TupleArrayConversions => [TUPLE_ARRAY_CONVERSIONS]);

pub struct TupleArrayConversions {
    msrv: Msrv,
}
impl TupleArrayConversions {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl LateLintPass<'_> for TupleArrayConversions {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.in_external_macro(cx.sess().source_map()) || !self.msrv.meets(cx, msrvs::TUPLE_ARRAY_CONVERSIONS) {
            return;
        }

        match expr.kind {
            ExprKind::Array(elements) if (1..=12).contains(&elements.len()) => check_array(cx, expr, elements),
            ExprKind::Tup(elements) if (1..=12).contains(&elements.len()) => check_tuple(cx, expr, elements),
            _ => {},
        }
    }
}

fn check_array<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, elements: &'tcx [Expr<'tcx>]) {
    let Some(ty) = cx.typeck_results().expr_ty(expr).builtin_index() else {
        unreachable!("`expr` must be an array or slice due to `ExprKind::Array`");
    };

    if let [first, ..] = elements
        && let Some(locals) = (match first.kind {
            ExprKind::Field(_, _) => elements
                .iter()
                .enumerate()
                .map(|(i, f)| -> Option<&'tcx Expr<'tcx>> {
                    let ExprKind::Field(lhs, ident) = f.kind else {
                        return None;
                    };
                    (ident.name.as_str() == i.to_string()).then_some(lhs)
                })
                .collect::<Option<Vec<_>>>(),
            ExprKind::Path(_) => Some(elements.iter().collect()),
            _ => None,
        })
        && all_bindings_are_for_conv(cx, &[ty], expr, elements, &locals, ToType::Array)
        && !is_from_proc_macro(cx, expr)
    {
        span_lint_and_help(
            cx,
            TUPLE_ARRAY_CONVERSIONS,
            expr.span,
            "it looks like you're trying to convert a tuple to an array",
            None,
            "use `.into()` instead, or `<[T; N]>::from` if type annotations are needed",
        );
    }
}

fn check_tuple<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, elements: &'tcx [Expr<'tcx>]) {
    if let ty::Tuple(tys) = cx.typeck_results().expr_ty(expr).kind()
        && let [first, ..] = elements
        // Fix #11100
        && tys.iter().all_equal()
        && let Some(locals) = (match first.kind {
            ExprKind::Index(..) => elements
                .iter()
                .enumerate()
                .map(|(i, i_expr)| -> Option<&'tcx Expr<'tcx>> {
                    if let ExprKind::Index(lhs, index, _) = i_expr.kind
                        && let ExprKind::Lit(lit) = index.kind
                        && let LitKind::Int(val, _) = lit.node
                    {
                        return (val == i as u128).then_some(lhs);
                    }

                    None
                })
                .collect::<Option<Vec<_>>>(),
            ExprKind::Path(_) => Some(elements.iter().collect()),
            _ => None,
        })
        && all_bindings_are_for_conv(cx, tys, expr, elements, &locals, ToType::Tuple)
        && !is_from_proc_macro(cx, expr)
    {
        span_lint_and_help(
            cx,
            TUPLE_ARRAY_CONVERSIONS,
            expr.span,
            "it looks like you're trying to convert an array to a tuple",
            None,
            "use `.into()` instead, or `<(T0, T1, ..., Tn)>::from` if type annotations are needed",
        );
    }
}

/// Checks that every binding in `elements` comes from the same parent `Pat` with the kind if there
/// is a parent `Pat`. Returns false in any of the following cases:
/// * `kind` does not match `pat.kind`
/// * one or more elements in `elements` is not a binding
/// * one or more bindings does not have the same parent `Pat`
/// * one or more bindings are used after `expr`
/// * the bindings do not all have the same type
#[expect(clippy::cast_possible_truncation)]
fn all_bindings_are_for_conv<'tcx>(
    cx: &LateContext<'tcx>,
    final_tys: &[Ty<'tcx>],
    expr: &Expr<'_>,
    elements: &[Expr<'_>],
    locals: &[&Expr<'_>],
    kind: ToType,
) -> bool {
    let Some(locals) = locals.iter().map(|e| path_to_local(e)).collect::<Option<Vec<_>>>() else {
        return false;
    };
    let local_parents = locals.iter().map(|l| cx.tcx.parent_hir_node(*l)).collect::<Vec<_>>();

    local_parents
        .iter()
        .map(|node| match node {
            Node::Pat(pat) => kind.eq(&pat.kind).then_some(pat.hir_id),
            Node::LetStmt(l) => Some(l.hir_id),
            _ => None,
        })
        .all_equal()
        // Fix #11124, very convenient utils function! ❤️
        && locals
            .iter()
            .all(|&l| for_each_local_use_after_expr(cx, l, expr.hir_id, |_| ControlFlow::Break::<()>(())).is_continue())
        && local_parents.first().is_some_and(|node| {
            let Some(ty) = match node {
                Node::Pat(pat) => Some(pat.hir_id),
                Node::LetStmt(l) => Some(l.hir_id),
                _ => None,
            }
            .map(|hir_id| cx.typeck_results().node_type(hir_id)) else {
                return false;
            };
            match (kind, ty.kind()) {
                // Ensure the final type and the original type have the same length, and that there
                // is no implicit `&mut`<=>`&` anywhere (#11100). Bit ugly, I know, but it works.
                (ToType::Array, ty::Tuple(tys)) => {
                    tys.len() == elements.len() && tys.iter().chain(final_tys.iter().copied()).all_equal()
                },
                (ToType::Tuple, ty::Array(ty, len)) => {
                    let Some(len) = len.try_to_target_usize(cx.tcx) else { return false };
                    len as usize == elements.len() && final_tys.iter().chain(once(ty)).all_equal()
                },
                _ => false,
            }
        })
}

#[derive(Clone, Copy)]
enum ToType {
    Array,
    Tuple,
}

impl PartialEq<PatKind<'_>> for ToType {
    fn eq(&self, other: &PatKind<'_>) -> bool {
        match self {
            ToType::Array => matches!(other, PatKind::Tuple(_, _)),
            ToType::Tuple => matches!(other, PatKind::Slice(_, _, _)),
        }
    }
}
