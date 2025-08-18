use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{FormatArgsStorage, find_format_arg_expr, is_format_macro, root_macro_call_first_node};
use clippy_utils::source::{indent_of, reindent_multiline, snippet_with_context};
use clippy_utils::visitors::{for_each_local_assignment, for_each_value_source};
use core::ops::ControlFlow;
use rustc_ast::{FormatArgs, FormatArgumentKind};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_body, walk_expr};
use rustc_hir::{Expr, ExprKind, HirId, HirIdSet, LetStmt, MatchSource, Node, PatKind, QPath, TyKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty;
use rustc_span::Span;

use super::LET_UNIT_VALUE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, format_args: &FormatArgsStorage, local: &'tcx LetStmt<'_>) {
    // skip `let () = { ... }`
    if let PatKind::Tuple(fields, ..) = local.pat.kind
        && fields.is_empty()
    {
        return;
    }

    if let Some(init) = local.init
        && !local.pat.span.from_expansion()
        && !local.span.in_external_macro(cx.sess().source_map())
        && !local.span.is_from_async_await()
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

        if (local.ty.is_some_and(|ty| !matches!(ty.kind, TyKind::Infer(())))
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
                    let mut suggestions = Vec::new();

                    // Suggest omitting the `let` binding
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, init.span, local.span.ctxt(), "()", &mut app).0;

                    // If this is a binding pattern, we need to add suggestions to remove any usages
                    // of the variable
                    if let PatKind::Binding(_, binding_hir_id, ..) = local.pat.kind
                        && let Some(body_id) = cx.enclosing_body.as_ref()
                    {
                        let body = cx.tcx.hir_body(*body_id);
                        let mut visitor = UnitVariableCollector::new(cx, format_args, binding_hir_id);
                        walk_body(&mut visitor, body);

                        let mut has_in_format_capture = false;
                        suggestions.extend(visitor.spans.iter().filter_map(|span| match span {
                            MaybeInFormatCapture::Yes => {
                                has_in_format_capture = true;
                                None
                            },
                            MaybeInFormatCapture::No(span) => Some((*span, "()".to_string())),
                        }));

                        if has_in_format_capture {
                            suggestions.push((
                                init.span,
                                format!("();\n{}", reindent_multiline(&snip, false, indent_of(cx, local.span))),
                            ));
                            diag.multipart_suggestion(
                                "replace variable usages with `()`",
                                suggestions,
                                Applicability::MachineApplicable,
                            );
                            return;
                        }
                    }

                    suggestions.push((local.span, format!("{snip};")));
                    let message = if suggestions.len() == 1 {
                        "omit the `let` binding"
                    } else {
                        "omit the `let` binding and replace variable usages with `()`"
                    };
                    diag.multipart_suggestion(message, suggestions, Applicability::MachineApplicable);
                },
            );
        }
    }
}

struct UnitVariableCollector<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    format_args: &'a FormatArgsStorage,
    id: HirId,
    spans: Vec<MaybeInFormatCapture>,
    macro_call: Option<&'a FormatArgs>,
}

enum MaybeInFormatCapture {
    Yes,
    No(Span),
}

impl<'a, 'tcx> UnitVariableCollector<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, format_args: &'a FormatArgsStorage, id: HirId) -> Self {
        Self {
            cx,
            format_args,
            id,
            spans: Vec::new(),
            macro_call: None,
        }
    }
}

/**
 * Collect all instances where a variable is used based on its `HirId`.
 */
impl<'tcx> Visitor<'tcx> for UnitVariableCollector<'_, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) -> Self::Result {
        if let Some(macro_call) = root_macro_call_first_node(self.cx, ex)
            && is_format_macro(self.cx, macro_call.def_id)
            && let Some(format_args) = self.format_args.get(self.cx, ex, macro_call.expn)
        {
            let parent_macro_call = self.macro_call;
            self.macro_call = Some(format_args);
            walk_expr(self, ex);
            self.macro_call = parent_macro_call;
            return;
        }

        if let ExprKind::Path(QPath::Resolved(None, path)) = ex.kind
            && let Res::Local(id) = path.res
            && id == self.id
        {
            if let Some(macro_call) = self.macro_call
                && macro_call.arguments.all_args().iter().any(|arg| {
                    matches!(arg.kind, FormatArgumentKind::Captured(_)) && find_format_arg_expr(ex, arg).is_some()
                })
            {
                self.spans.push(MaybeInFormatCapture::Yes);
            } else {
                self.spans.push(MaybeInFormatCapture::No(path.span));
            }
        }

        walk_expr(self, ex);
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
            if !l.ty.is_none_or(|ty| matches!(ty.kind, TyKind::Infer(()))) {
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
                kind: ExprKind::Path(path),
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
