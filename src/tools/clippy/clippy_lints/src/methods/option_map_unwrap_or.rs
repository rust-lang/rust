use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{Visitor, walk_path};
use rustc_hir::{ExprKind, HirId, Node, PatKind, Path, QPath};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_span::{Span, sym};
use std::ops::ControlFlow;

use super::MAP_UNWRAP_OR;

/// lint use of `map().unwrap_or()` for `Option`s
#[expect(clippy::too_many_arguments)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &rustc_hir::Expr<'_>,
    recv: &rustc_hir::Expr<'_>,
    map_arg: &'tcx rustc_hir::Expr<'_>,
    unwrap_recv: &rustc_hir::Expr<'_>,
    unwrap_arg: &'tcx rustc_hir::Expr<'_>,
    map_span: Span,
    msrv: Msrv,
) {
    // lint if the caller of `map()` is an `Option`
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option) {
        if !is_copy(cx, cx.typeck_results().expr_ty(unwrap_arg)) {
            // Replacing `.map(<f>).unwrap_or(<a>)` with `.map_or(<a>, <f>)` can sometimes lead to
            // borrowck errors, see #10579 for one such instance.
            // In particular, if `a` causes a move and `f` references that moved binding, then we cannot lint:
            // ```
            // let x = vec![1, 2];
            // x.get(0..1).map(|s| s.to_vec()).unwrap_or(x);
            // ```
            // This compiles, but changing it to `map_or` will produce a compile error:
            // ```
            // let x = vec![1, 2];
            // x.get(0..1).map_or(x, |s| s.to_vec())
            //                    ^ moving `x` here
            // ^^^^^^^^^^^ while it is borrowed here (and later used in the closure)
            // ```
            // So, we have to check that `a` is not referenced anywhere (even outside of the `.map` closure!)
            // before the call to `unwrap_or`.

            let mut unwrap_visitor = UnwrapVisitor {
                cx,
                identifiers: FxHashSet::default(),
            };
            unwrap_visitor.visit_expr(unwrap_arg);

            let mut reference_visitor = ReferenceVisitor {
                cx,
                identifiers: unwrap_visitor.identifiers,
                unwrap_or_span: unwrap_arg.span,
            };

            let body = cx.tcx.hir_body_owned_by(cx.tcx.hir_enclosing_body_owner(expr.hir_id));

            // Visit the body, and return if we've found a reference
            if reference_visitor.visit_body(body).is_break() {
                return;
            }
        }

        if !unwrap_arg.span.eq_ctxt(map_span) {
            return;
        }

        // is_some_and is stabilised && `unwrap_or` argument is false; suggest `is_some_and` instead
        let suggest_is_some_and = matches!(&unwrap_arg.kind, ExprKind::Lit(lit)
            if matches!(lit.node, rustc_ast::LitKind::Bool(false)))
            && msrv.meets(cx, msrvs::OPTION_RESULT_IS_VARIANT_AND);

        let mut applicability = Applicability::MachineApplicable;
        // get snippet for unwrap_or()
        let unwrap_snippet = snippet_with_applicability(cx, unwrap_arg.span, "..", &mut applicability);
        // lint message
        // comparing the snippet from source to raw text ("None") below is safe
        // because we already have checked the type.
        let arg = if unwrap_snippet == "None" {
            "None"
        } else if suggest_is_some_and {
            "false"
        } else {
            "<a>"
        };
        let unwrap_snippet_none = unwrap_snippet == "None";
        let suggest = if unwrap_snippet_none {
            "and_then(<f>)"
        } else if suggest_is_some_and {
            "is_some_and(<f>)"
        } else {
            "map_or(<a>, <f>)"
        };
        let msg = format!("called `map(<f>).unwrap_or({arg})` on an `Option` value");

        span_lint_and_then(cx, MAP_UNWRAP_OR, expr.span, msg, |diag| {
            let map_arg_span = map_arg.span;

            let mut suggestion = vec![
                (
                    map_span,
                    String::from(if unwrap_snippet_none {
                        "and_then"
                    } else if suggest_is_some_and {
                        "is_some_and"
                    } else {
                        "map_or"
                    }),
                ),
                (expr.span.with_lo(unwrap_recv.span.hi()), String::new()),
            ];

            if !unwrap_snippet_none && !suggest_is_some_and {
                suggestion.push((map_arg_span.with_hi(map_arg_span.lo()), format!("{unwrap_snippet}, ")));
            }

            diag.multipart_suggestion(format!("use `{suggest}` instead"), suggestion, applicability);
        });
    }
}

struct UnwrapVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    identifiers: FxHashSet<HirId>,
}

impl<'tcx> Visitor<'tcx> for UnwrapVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_path(&mut self, path: &Path<'tcx>, _: HirId) {
        if let Res::Local(local_id) = path.res
            && let Node::Pat(pat) = self.cx.tcx.hir_node(local_id)
            && let PatKind::Binding(_, local_id, ..) = pat.kind
        {
            self.identifiers.insert(local_id);
        }
        walk_path(self, path);
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

struct ReferenceVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    identifiers: FxHashSet<HirId>,
    unwrap_or_span: Span,
}

impl<'tcx> Visitor<'tcx> for ReferenceVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::All;
    type Result = ControlFlow<()>;
    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'_>) -> ControlFlow<()> {
        // If we haven't found a reference yet, check if this references
        // one of the locals that was moved in the `unwrap_or` argument.
        // We are only interested in exprs that appear before the `unwrap_or` call.
        if expr.span < self.unwrap_or_span
            && let ExprKind::Path(ref path) = expr.kind
            && let QPath::Resolved(_, path) = path
            && let Res::Local(local_id) = path.res
            && let Node::Pat(pat) = self.cx.tcx.hir_node(local_id)
            && let PatKind::Binding(_, local_id, ..) = pat.kind
            && self.identifiers.contains(&local_id)
        {
            return ControlFlow::Break(());
        }
        rustc_hir::intravisit::walk_expr(self, expr)
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}
