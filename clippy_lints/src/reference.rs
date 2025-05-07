use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, snippet_with_applicability};
use clippy_utils::ty::adjust_derefs_manually_drop;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId, Mutability, Node, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `*&` and `*&mut` in expressions.
    ///
    /// ### Why is this bad?
    /// Immediately dereferencing a reference is no-op and
    /// makes the code less clear.
    ///
    /// ### Known problems
    /// Multiple dereference/addrof pairs are not handled so
    /// the suggested fix for `x = **&&y` is `x = *&y`, which is still incorrect.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a = f(*&mut b);
    /// let c = *&d;
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let a = f(b);
    /// let c = d;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DEREF_ADDROF,
    complexity,
    "use of `*&` or `*&mut` in an expression"
}

declare_lint_pass!(DerefAddrOf => [DEREF_ADDROF]);

impl LateLintPass<'_> for DerefAddrOf {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        if let ExprKind::Unary(UnOp::Deref, deref_target) = e.kind
            && let ExprKind::AddrOf(_, mutability, addrof_target) = deref_target.kind
            // NOTE(tesuji): `*&` forces rustc to const-promote the array to `.rodata` section.
            // See #12854 for details.
            && !matches!(addrof_target.kind, ExprKind::Array(_))
            && deref_target.span.eq_ctxt(e.span)
            && !addrof_target.span.from_expansion()
        {
            // If this expression is an explicit `DerefMut` of a `ManuallyDrop` reached through a
            // union, we may remove the reference if we are at the point where the implicit
            // dereference would take place. Otherwise, we should not lint.
            let keep_deref = match is_manually_drop_through_union(cx, e.hir_id, addrof_target) {
                ManuallyDropThroughUnion::Directly => true,
                ManuallyDropThroughUnion::Indirect => return,
                ManuallyDropThroughUnion::No => false,
            };

            let mut applicability = Applicability::MachineApplicable;
            let sugg = if e.span.from_expansion() {
                if let Some(macro_source) = e.span.get_source_text(cx) {
                    // Remove leading whitespace from the given span
                    // e.g: ` $visitor` turns into `$visitor`
                    let trim_leading_whitespaces = |span: Span| {
                        span.get_source_text(cx)
                            .and_then(|snip| {
                                #[expect(clippy::cast_possible_truncation)]
                                snip.find(|c: char| !c.is_whitespace())
                                    .map(|pos| span.lo() + BytePos(pos as u32))
                            })
                            .map_or(span, |start_no_whitespace| e.span.with_lo(start_no_whitespace))
                    };

                    let mut generate_snippet = |pattern: &str| {
                        #[expect(clippy::cast_possible_truncation)]
                        macro_source.rfind(pattern).map(|pattern_pos| {
                            let rpos = pattern_pos + pattern.len();
                            let span_after_ref = e.span.with_lo(BytePos(e.span.lo().0 + rpos as u32));
                            let span = trim_leading_whitespaces(span_after_ref);
                            snippet_with_applicability(cx, span, "_", &mut applicability)
                        })
                    };

                    if mutability == Mutability::Mut {
                        generate_snippet("mut")
                    } else {
                        generate_snippet("&")
                    }
                } else {
                    Some(snippet_with_applicability(cx, e.span, "_", &mut applicability))
                }
            } else {
                Some(snippet_with_applicability(
                    cx,
                    addrof_target.span,
                    "_",
                    &mut applicability,
                ))
            };
            if let Some(sugg) = sugg {
                span_lint_and_sugg(
                    cx,
                    DEREF_ADDROF,
                    e.span,
                    "immediately dereferencing a reference",
                    "try",
                    if keep_deref {
                        format!("(*{sugg})")
                    } else {
                        sugg.to_string()
                    },
                    applicability,
                );
            }
        }
    }
}

/// Is this a `ManuallyDrop` reached through a union, and when is `DerefMut` called on it?
enum ManuallyDropThroughUnion {
    /// `ManuallyDrop` reached through a union and immediately explicitely dereferenced
    Directly,
    /// `ManuallyDrop` reached through a union, and dereferenced later on
    Indirect,
    /// Any other situation
    No,
}

/// Check if `addrof_target` is part of an access to a `ManuallyDrop` entity reached through a
/// union, and when it is dereferenced using `DerefMut` starting from `expr_id` and going up.
fn is_manually_drop_through_union(
    cx: &LateContext<'_>,
    expr_id: HirId,
    addrof_target: &Expr<'_>,
) -> ManuallyDropThroughUnion {
    if is_reached_through_union(cx, addrof_target) {
        let typeck = cx.typeck_results();
        for (idx, id) in std::iter::once(expr_id)
            .chain(cx.tcx.hir_parent_id_iter(expr_id))
            .enumerate()
        {
            if let Node::Expr(expr) = cx.tcx.hir_node(id) {
                if adjust_derefs_manually_drop(typeck.expr_adjustments(expr), typeck.expr_ty(expr)) {
                    return if idx == 0 {
                        ManuallyDropThroughUnion::Directly
                    } else {
                        ManuallyDropThroughUnion::Indirect
                    };
                }
            } else {
                break;
            }
        }
    }
    ManuallyDropThroughUnion::No
}

/// Checks whether `expr` denotes an object reached through a union
fn is_reached_through_union(cx: &LateContext<'_>, mut expr: &Expr<'_>) -> bool {
    while let ExprKind::Field(parent, _) | ExprKind::Index(parent, _, _) = expr.kind {
        if cx.typeck_results().expr_ty_adjusted(parent).is_union() {
            return true;
        }
        expr = parent;
    }
    false
}
