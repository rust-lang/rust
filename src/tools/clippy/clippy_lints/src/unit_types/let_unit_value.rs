use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::{for_each_local_assignment, for_each_value_source, is_local_used};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_body};
use rustc_hir::{Expr, ExprKind, HirId, HirIdSet, LetStmt, MatchSource, Node, PatKind, QPath, TyKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::{in_external_macro, is_from_async_await};
use rustc_middle::ty;

use super::LET_UNIT_VALUE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, local: &'tcx LetStmt<'_>) {
    // skip `let () = { ... }`
    if let PatKind::Tuple(fields, ..) = local.pat.kind
        && fields.is_empty()
    {
        return;
    }

    if let Some(init) = local.init
        && !local.pat.span.from_expansion()
        && !in_external_macro(cx.sess(), local.span)
        && !is_from_async_await(local.span)
        && cx.typeck_results().pat_ty(local.pat).is_unit()
    {
        // skip `let awa = ()`
        if let ExprKind::Tup([]) = init.kind {
            return;
        }

        // skip `let _: () = { ... }`
        if let Some(ty) = local.ty
            && let TyKind::Tup([]) = ty.kind
        {
            return;
        }

        if (local.ty.map_or(false, |ty| !matches!(ty.kind, TyKind::Infer))
            || matches!(local.pat.kind, PatKind::Tuple([], ddpos) if ddpos.as_opt_usize().is_none()))
            && expr_needs_inferred_result(cx, init)
        {
            if !matches!(local.pat.kind, PatKind::Wild)
                && !matches!(local.pat.kind, PatKind::Tuple([], ddpos) if ddpos.as_opt_usize().is_none())
            {
                span_lint_and_then(
                    cx,
                    LET_UNIT_VALUE,
                    local.span,
                    "this let-binding has unit value",
                    |diag| {
                        diag.span_suggestion(
                            local.pat.span,
                            "use a wildcard binding",
                            "_",
                            Applicability::MaybeIncorrect, // snippet
                        );
                    },
                );
            }
        } else {
            if let ExprKind::Match(_, _, MatchSource::AwaitDesugar) = init.kind {
                return;
            }

            span_lint_and_then(
                cx,
                LET_UNIT_VALUE,
                local.span,
                "this let-binding has unit value",
                |diag| {
                    if let Some(expr) = &local.init {
                        let mut app = Applicability::MachineApplicable;
                        let snip = snippet_with_context(cx, expr.span, local.span.ctxt(), "()", &mut app).0;
                        diag.span_suggestion(local.span, "omit the `let` binding", format!("{snip};"), app);
                    }

                    if let PatKind::Binding(_, binding_hir_id, ident, ..) = local.pat.kind
                        && let Some(body_id) = cx.enclosing_body.as_ref()
                        && let body = cx.tcx.hir().body(*body_id)
                        && is_local_used(cx, body, binding_hir_id)
                    {
                        let identifier = ident.as_str();
                        let mut visitor = UnitVariableCollector::new(binding_hir_id);
                        walk_body(&mut visitor, body);
                        visitor.spans.into_iter().for_each(|span| {
                            let msg =
                                format!("variable `{identifier}` of type `()` can be replaced with explicit `()`");
                            diag.span_suggestion(span, msg, "()", Applicability::MachineApplicable);
                        });
                    }
                },
            );
        }
    }
}

struct UnitVariableCollector {
    id: HirId,
    spans: Vec<rustc_span::Span>,
}

impl UnitVariableCollector {
    fn new(id: HirId) -> Self {
        Self { id, spans: vec![] }
    }
}

/**
 * Collect all instances where a variable is used based on its `HirId`.
 */
impl<'tcx> Visitor<'tcx> for UnitVariableCollector {
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) -> Self::Result {
        if let ExprKind::Path(QPath::Resolved(None, path)) = ex.kind
            && let Res::Local(id) = path.res
            && id == self.id
        {
            self.spans.push(path.span);
        }
        rustc_hir::intravisit::walk_expr(self, ex);
    }
}

/// Checks sub-expressions which create the value returned by the given expression for whether
/// return value inference is needed. This checks through locals to see if they also need inference
/// at this point.
///
/// e.g.
/// ```rust,ignore
/// let bar = foo();
/// let x: u32 = if true { baz() } else { bar };
/// ```
/// Here the sources of the value assigned to `x` would be `baz()`, and `foo()` via the
/// initialization of `bar`. If both `foo` and `baz` have a return type which require type
/// inference then this function would return `true`.
fn expr_needs_inferred_result<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> bool {
    // The locals used for initialization which have yet to be checked.
    let mut locals_to_check = Vec::new();
    // All the locals which have been added to `locals_to_check`. Needed to prevent cycles.
    let mut seen_locals = HirIdSet::default();
    if !each_value_source_needs_inference(cx, e, &mut locals_to_check, &mut seen_locals) {
        return false;
    }
    while let Some(id) = locals_to_check.pop() {
        if let Node::LetStmt(l) = cx.tcx.parent_hir_node(id) {
            if !l.ty.map_or(true, |ty| matches!(ty.kind, TyKind::Infer)) {
                return false;
            }
            if let Some(e) = l.init {
                if !each_value_source_needs_inference(cx, e, &mut locals_to_check, &mut seen_locals) {
                    return false;
                }
            } else if for_each_local_assignment(cx, id, |e| {
                if each_value_source_needs_inference(cx, e, &mut locals_to_check, &mut seen_locals) {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            })
            .is_break()
            {
                return false;
            }
        }
    }

    true
}

fn each_value_source_needs_inference(
    cx: &LateContext<'_>,
    e: &Expr<'_>,
    locals_to_check: &mut Vec<HirId>,
    seen_locals: &mut HirIdSet,
) -> bool {
    for_each_value_source(e, &mut |e| {
        if needs_inferred_result_ty(cx, e, locals_to_check, seen_locals) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    })
    .is_continue()
}

fn needs_inferred_result_ty(
    cx: &LateContext<'_>,
    e: &Expr<'_>,
    locals_to_check: &mut Vec<HirId>,
    seen_locals: &mut HirIdSet,
) -> bool {
    let (id, receiver, args) = match e.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(ref path),
                hir_id,
                ..
            },
            args,
        ) => match cx.qpath_res(path, *hir_id) {
            Res::Def(DefKind::AssocFn | DefKind::Fn, id) => (id, None, args),
            _ => return false,
        },
        ExprKind::MethodCall(_, receiver, args, _) => match cx.typeck_results().type_dependent_def_id(e.hir_id) {
            Some(id) => (id, Some(receiver), args),
            None => return false,
        },
        ExprKind::Path(QPath::Resolved(None, path)) => {
            if let Res::Local(id) = path.res
                && seen_locals.insert(id)
            {
                locals_to_check.push(id);
            }
            return true;
        },
        _ => return false,
    };
    let sig = cx.tcx.fn_sig(id).instantiate_identity().skip_binder();
    if let ty::Param(output_ty) = *sig.output().kind() {
        let args: Vec<&Expr<'_>> = if let Some(receiver) = receiver {
            std::iter::once(receiver).chain(args.iter()).collect()
        } else {
            args.iter().collect()
        };
        sig.inputs().iter().zip(args).all(|(&ty, arg)| {
            !ty.is_param(output_ty.index) || each_value_source_needs_inference(cx, arg, locals_to_check, seen_locals)
        })
    } else {
        false
    }
}
