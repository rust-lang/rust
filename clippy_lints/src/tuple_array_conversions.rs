use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::{is_from_proc_macro, path_to_local};
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind, HirId, Node, Pat};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use std::iter::once;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for tuple<=>array conversions that are not done with `.into()`.
    ///
    /// ### Why is this bad?
    /// It may be unnecessary complexity. `.into()` works for converting tuples
    /// <=> arrays of up to 12 elements and may convey intent more clearly.
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
    pedantic,
    "checks for tuple<=>array conversions that are not done with `.into()`"
}
impl_lint_pass!(TupleArrayConversions => [TUPLE_ARRAY_CONVERSIONS]);

#[derive(Clone)]
pub struct TupleArrayConversions {
    pub msrv: Msrv,
}

impl LateLintPass<'_> for TupleArrayConversions {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !in_external_macro(cx.sess(), expr.span) && self.msrv.meets(msrvs::TUPLE_ARRAY_CONVERSIONS) {
            match expr.kind {
                ExprKind::Array(elements) if (1..=12).contains(&elements.len()) => check_array(cx, expr, elements),
                ExprKind::Tup(elements) if (1..=12).contains(&elements.len()) => check_tuple(cx, expr, elements),
                _ => {},
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

#[expect(
    clippy::blocks_in_if_conditions,
    reason = "not a FP, but this is much easier to understand"
)]
fn check_array<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, elements: &'tcx [Expr<'tcx>]) {
    if should_lint(
        cx,
        elements,
        // This is cursed.
        Some,
        |(first_id, local)| {
            if let Node::Pat(pat) = local
                && let parent = parent_pat(cx, pat)
                && parent.hir_id == first_id
            {
                return matches!(
                    cx.typeck_results().pat_ty(parent).peel_refs().kind(),
                    ty::Tuple(len) if len.len() == elements.len()
                );
            }

            false
        },
    ) || should_lint(
        cx,
        elements,
        |(i, expr)| {
            if let ExprKind::Field(path, field) = expr.kind && field.as_str() == i.to_string() {
                return Some((i, path));
            };

            None
        },
        |(first_id, local)| {
            if let Node::Pat(pat) = local
                && let parent = parent_pat(cx, pat)
                && parent.hir_id == first_id
            {
                return matches!(
                    cx.typeck_results().pat_ty(parent).peel_refs().kind(),
                    ty::Tuple(len) if len.len() == elements.len()
                );
            }

            false
        },
    ) {
        emit_lint(cx, expr, ToType::Array);
    }
}

#[expect(
    clippy::blocks_in_if_conditions,
    reason = "not a FP, but this is much easier to understand"
)]
#[expect(clippy::cast_possible_truncation)]
fn check_tuple<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, elements: &'tcx [Expr<'tcx>]) {
    if should_lint(cx, elements, Some, |(first_id, local)| {
        if let Node::Pat(pat) = local
                && let parent = parent_pat(cx, pat)
                && parent.hir_id == first_id
            {
                return matches!(
                    cx.typeck_results().pat_ty(parent).peel_refs().kind(),
                    ty::Array(_, len) if len.eval_target_usize(cx.tcx, cx.param_env) as usize == elements.len()
                );
            }

        false
    }) || should_lint(
        cx,
        elements,
        |(i, expr)| {
            if let ExprKind::Index(path, index) = expr.kind
                && let ExprKind::Lit(lit) = index.kind
                && let LitKind::Int(val, _) = lit.node
                && val as usize == i
            {
                return Some((i, path));
            };

            None
        },
        |(first_id, local)| {
            if let Node::Pat(pat) = local
                && let parent = parent_pat(cx, pat)
                && parent.hir_id == first_id
            {
                return matches!(
                    cx.typeck_results().pat_ty(parent).peel_refs().kind(),
                    ty::Array(_, len) if len.eval_target_usize(cx.tcx, cx.param_env) as usize == elements.len()
                );
            }

            false
        },
    ) {
        emit_lint(cx, expr, ToType::Tuple);
    }
}

/// Walks up the `Pat` until it's reached the final containing `Pat`.
fn parent_pat<'tcx>(cx: &LateContext<'tcx>, start: &'tcx Pat<'tcx>) -> &'tcx Pat<'tcx> {
    let mut end = start;
    for (_, node) in cx.tcx.hir().parent_iter(start.hir_id) {
        if let Node::Pat(pat) = node {
            end = pat;
        } else {
            break;
        }
    }
    end
}

#[derive(Clone, Copy)]
enum ToType {
    Array,
    Tuple,
}

impl ToType {
    fn msg(self) -> &'static str {
        match self {
            ToType::Array => "it looks like you're trying to convert a tuple to an array",
            ToType::Tuple => "it looks like you're trying to convert an array to a tuple",
        }
    }

    fn help(self) -> &'static str {
        match self {
            ToType::Array => "use `.into()` instead, or `<[T; N]>::from` if type annotations are needed",
            ToType::Tuple => "use `.into()` instead, or `<(T0, T1, ..., Tn)>::from` if type annotations are needed",
        }
    }
}

fn emit_lint<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, to_type: ToType) -> bool {
    if !is_from_proc_macro(cx, expr) {
        span_lint_and_help(
            cx,
            TUPLE_ARRAY_CONVERSIONS,
            expr.span,
            to_type.msg(),
            None,
            to_type.help(),
        );

        return true;
    }

    false
}

fn should_lint<'tcx>(
    cx: &LateContext<'tcx>,
    elements: &'tcx [Expr<'tcx>],
    map: impl FnMut((usize, &'tcx Expr<'tcx>)) -> Option<(usize, &Expr<'_>)>,
    predicate: impl FnMut((HirId, &Node<'tcx>)) -> bool,
) -> bool {
    if let Some(elements) = elements
            .iter()
            .enumerate()
            .map(map)
            .collect::<Option<Vec<_>>>()
        && let Some(locals) = elements
            .iter()
            .map(|(_, element)| path_to_local(element).and_then(|local| cx.tcx.hir().find(local)))
            .collect::<Option<Vec<_>>>()
        && let [first, rest @ ..] = &*locals
        && let Node::Pat(first_pat) = first
        && let parent = parent_pat(cx, first_pat).hir_id
        && rest.iter().chain(once(first)).map(|i| (parent, i)).all(predicate)
    {
        return true;
    }

    false
}
