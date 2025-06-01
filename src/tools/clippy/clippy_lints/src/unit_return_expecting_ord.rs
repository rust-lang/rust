use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use rustc_hir::def_id::DefId;
use rustc_hir::{Closure, Expr, ExprKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::{ClauseKind, GenericPredicates, ProjectionPredicate, TraitPredicate};
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Span, Symbol, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that expect closures of type
    /// Fn(...) -> Ord where the implemented closure returns the unit type.
    /// The lint also suggests to remove the semi-colon at the end of the statement if present.
    ///
    /// ### Why is this bad?
    /// Likely, returning the unit type is unintentional, and
    /// could simply be caused by an extra semi-colon. Since () implements Ord
    /// it doesn't cause a compilation error.
    /// This is the same reasoning behind the unit_cmp lint.
    ///
    /// ### Known problems
    /// If returning unit is intentional, then there is no
    /// way of specifying this without triggering needless_return lint
    ///
    /// ### Example
    /// ```no_run
    /// let mut twins = vec![(1, 1), (2, 2)];
    /// twins.sort_by_key(|x| { x.1; });
    /// ```
    #[clippy::version = "1.47.0"]
    pub UNIT_RETURN_EXPECTING_ORD,
    correctness,
    "fn arguments of type Fn(...) -> Ord returning the unit type ()."
}

declare_lint_pass!(UnitReturnExpectingOrd => [UNIT_RETURN_EXPECTING_ORD]);

// For each
fn get_trait_predicates_for_trait_ids<'tcx>(
    cx: &LateContext<'tcx>,
    generics: GenericPredicates<'tcx>,
    trait_ids: &[Option<DefId>], // At least 2 ids
) -> [Vec<TraitPredicate<'tcx>>; 3] {
    debug_assert!(trait_ids.len() >= 2);
    let mut preds = [Vec::new(), Vec::new(), Vec::new()];
    for (pred, _) in generics.predicates {
        if let ClauseKind::Trait(poly_trait_pred) = pred.kind().skip_binder() {
            let trait_pred = cx
                .tcx
                .instantiate_bound_regions_with_erased(pred.kind().rebind(poly_trait_pred));
            for (i, tid) in trait_ids.iter().enumerate() {
                if let Some(tid) = tid
                    && *tid == trait_pred.trait_ref.def_id
                {
                    preds[i].push(trait_pred);
                }
            }
        }
    }
    preds
}

fn get_projection_pred<'tcx>(
    cx: &LateContext<'tcx>,
    generics: GenericPredicates<'tcx>,
    trait_pred: TraitPredicate<'tcx>,
) -> Option<ProjectionPredicate<'tcx>> {
    generics.predicates.iter().find_map(|(proj_pred, _)| {
        if let ClauseKind::Projection(pred) = proj_pred.kind().skip_binder() {
            let projection_pred = cx
                .tcx
                .instantiate_bound_regions_with_erased(proj_pred.kind().rebind(pred));
            if projection_pred.projection_term.args == trait_pred.trait_ref.args {
                return Some(projection_pred);
            }
        }
        None
    })
}

fn get_args_to_check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    args_len: usize,
    fn_mut_trait: DefId,
    ord_trait: Option<DefId>,
    partial_ord_trait: Option<DefId>,
) -> Vec<(usize, Symbol)> {
    let mut args_to_check = Vec::new();
    if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        let fn_sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let generics = cx.tcx.predicates_of(def_id);
        let [fn_mut_preds, ord_preds, partial_ord_preds] =
            get_trait_predicates_for_trait_ids(cx, generics, &[Some(fn_mut_trait), ord_trait, partial_ord_trait]);
        if fn_mut_preds.is_empty() {
            return vec![];
        }

        // Trying to call instantiate_bound_regions_with_erased on fn_sig.inputs() gives the following error
        // The trait `rustc::ty::TypeFoldable<'_>` is not implemented for
        // `&[rustc_middle::ty::Ty<'_>]`
        let inputs_output = cx.tcx.instantiate_bound_regions_with_erased(fn_sig.inputs_and_output());
        inputs_output
            .iter()
            .rev()
            .skip(1)
            .rev()
            .enumerate()
            .for_each(|(i, inp)| {
                for trait_pred in &fn_mut_preds {
                    if trait_pred.self_ty() == inp
                        && let Some(return_ty_pred) = get_projection_pred(cx, generics, *trait_pred)
                    {
                        if ord_preds
                            .iter()
                            .any(|ord| Some(ord.self_ty()) == return_ty_pred.term.as_type())
                        {
                            args_to_check.push((i, sym::Ord));
                            if args_to_check.len() == args_len - 1 {
                                break;
                            }
                        } else if partial_ord_preds
                            .iter()
                            .any(|pord| pord.self_ty() == return_ty_pred.term.expect_type())
                        {
                            args_to_check.push((i, sym::PartialOrd));
                            if args_to_check.len() == args_len - 1 {
                                break;
                            }
                        }
                    }
                }
            });
    }
    args_to_check
}

fn check_arg<'tcx>(cx: &LateContext<'tcx>, arg: &'tcx Expr<'tcx>) -> Option<(Span, Option<Span>)> {
    if let ExprKind::Closure(&Closure { body, fn_decl_span, .. }) = arg.kind
        && let ty::Closure(_def_id, args) = &cx.typeck_results().node_type(arg.hir_id).kind()
        && let ret_ty = args.as_closure().sig().output()
        && let ty = cx.tcx.instantiate_bound_regions_with_erased(ret_ty)
        && ty.is_unit()
    {
        let body = cx.tcx.hir_body(body);
        if let ExprKind::Block(block, _) = body.value.kind
            && block.expr.is_none()
            && let Some(stmt) = block.stmts.last()
            && let StmtKind::Semi(_) = stmt.kind
        {
            let data = stmt.span.data();
            // Make a span out of the semicolon for the help message
            Some((fn_decl_span, Some(data.with_lo(data.hi - BytePos(1)))))
        } else {
            Some((fn_decl_span, None))
        }
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for UnitReturnExpectingOrd {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::MethodCall(_, receiver, args, _) = expr.kind
            && args.iter().any(|arg| {
                matches!(
                    arg.peel_blocks().peel_borrows().peel_drop_temps().kind,
                    ExprKind::Path(_) | ExprKind::Closure(_)
                )
            })
            && let Some(fn_mut_trait) = cx.tcx.lang_items().fn_mut_trait()
        {
            let ord_trait = cx.tcx.get_diagnostic_item(sym::Ord);
            let partial_ord_trait = cx.tcx.lang_items().partial_ord_trait();
            if (ord_trait, partial_ord_trait) == (None, None) {
                return;
            }

            let args = std::iter::once(receiver).chain(args.iter()).collect::<Vec<_>>();
            let arg_indices = get_args_to_check(cx, expr, args.len(), fn_mut_trait, ord_trait, partial_ord_trait);
            for (i, trait_name) in arg_indices {
                match check_arg(cx, args[i]) {
                    Some((span, None)) => {
                        span_lint(
                            cx,
                            UNIT_RETURN_EXPECTING_ORD,
                            span,
                            format!(
                                "this closure returns \
                                   the unit type which also implements {trait_name}"
                            ),
                        );
                    },
                    Some((span, Some(last_semi))) => {
                        span_lint_and_help(
                            cx,
                            UNIT_RETURN_EXPECTING_ORD,
                            span,
                            format!(
                                "this closure returns \
                                   the unit type which also implements {trait_name}"
                            ),
                            Some(last_semi),
                            "probably caused by this trailing semicolon",
                        );
                    },
                    None => {},
                }
            }
        }
    }
}
