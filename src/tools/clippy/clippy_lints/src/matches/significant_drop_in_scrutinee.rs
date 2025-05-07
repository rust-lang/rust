use std::ops::ControlFlow;

use crate::FxHashSet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{first_line_of_span, indent_of, snippet};
use clippy_utils::ty::{for_each_top_level_late_bound_region, is_copy};
use clippy_utils::{get_attr, is_lint_allowed};
use itertools::Itertools;
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, Diag};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::{GenericArgKind, Region, RegionKind, Ty, TyCtxt, TypeVisitable, TypeVisitor};
use rustc_span::Span;

use super::SIGNIFICANT_DROP_IN_SCRUTINEE;

pub(super) fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'_>],
    source: MatchSource,
) {
    if is_lint_allowed(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, expr.hir_id) {
        return;
    }

    let scrutinee = match (source, &scrutinee.kind) {
        (MatchSource::ForLoopDesugar, ExprKind::Call(_, [e])) => e,
        _ => scrutinee,
    };

    let message = if source == MatchSource::Normal {
        "temporary with significant `Drop` in `match` scrutinee will live until the end of the `match` expression"
    } else {
        "temporary with significant `Drop` in `for` loop condition will live until the end of the `for` expression"
    };

    let arms = arms.iter().map(|arm| arm.body).collect::<Vec<_>>();

    check(cx, expr, scrutinee, &arms, message, Suggestion::Emit);
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    if_then: &'tcx Expr<'_>,
    if_else: Option<&'tcx Expr<'_>>,
) {
    if is_lint_allowed(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, expr.hir_id) {
        return;
    }

    let message =
        "temporary with significant `Drop` in `if let` scrutinee will live until the end of the `if let` expression";

    if let Some(if_else) = if_else {
        check(cx, expr, scrutinee, &[if_then, if_else], message, Suggestion::Emit);
    } else {
        check(cx, expr, scrutinee, &[if_then], message, Suggestion::Emit);
    }
}

pub(super) fn check_while_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
) {
    if is_lint_allowed(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, expr.hir_id) {
        return;
    }

    check(
        cx,
        expr,
        scrutinee,
        &[body],
        "temporary with significant `Drop` in `while let` scrutinee will live until the end of the `while let` expression",
        // Don't emit wrong suggestions: We cannot fix the significant drop in the `while let` scrutinee by simply
        // moving it out. We need to change the `while` to a `loop` instead.
        Suggestion::DontEmit,
    );
}

#[derive(Copy, Clone, Debug)]
enum Suggestion {
    Emit,
    DontEmit,
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    arms: &[&'tcx Expr<'_>],
    message: &'static str,
    sugg: Suggestion,
) {
    let mut helper = SigDropHelper::new(cx);
    let suggestions = helper.find_sig_drop(scrutinee);

    for found in suggestions {
        span_lint_and_then(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, found.found_span, message, |diag| {
            match sugg {
                Suggestion::Emit => set_suggestion(diag, cx, expr, found),
                Suggestion::DontEmit => (),
            }

            let s = Span::new(expr.span.hi(), expr.span.hi(), expr.span.ctxt(), None);
            diag.span_label(s, "temporary lives until here");
            for span in has_significant_drop_in_arms(cx, arms) {
                diag.span_label(span, "another value with significant `Drop` created here");
            }
            diag.note("this might lead to deadlocks or other unexpected behavior");
        });
    }
}

fn set_suggestion<'tcx>(diag: &mut Diag<'_, ()>, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, found: FoundSigDrop) {
    let original = snippet(cx, found.found_span, "..");
    let trailing_indent = " ".repeat(indent_of(cx, found.found_span).unwrap_or(0));

    let replacement = {
        let (def_part, deref_part) = if found.is_unit_return_val {
            ("", String::new())
        } else {
            ("let value = ", "*".repeat(found.peel_ref_times))
        };
        format!("{def_part}{deref_part}{original};\n{trailing_indent}")
    };

    let suggestion_message = if found.peel_ref_times == 0 {
        "try moving the temporary above the match"
    } else {
        "try moving the temporary above the match and create a copy"
    };

    let scrutinee_replacement = if found.is_unit_return_val {
        "()".to_owned()
    } else if found.peel_ref_times == 0 {
        "value".to_owned()
    } else {
        let ref_part = "&".repeat(found.peel_ref_times);
        format!("({ref_part}value)")
    };

    diag.multipart_suggestion(
        suggestion_message,
        vec![
            (first_line_of_span(cx, expr.span).shrink_to_lo(), replacement),
            (found.found_span, scrutinee_replacement),
        ],
        Applicability::MaybeIncorrect,
    );
}

struct SigDropChecker<'a, 'tcx> {
    seen_types: FxHashSet<Ty<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> SigDropChecker<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> SigDropChecker<'a, 'tcx> {
        SigDropChecker {
            seen_types: FxHashSet::default(),
            cx,
        }
    }

    fn is_sig_drop_expr(&mut self, ex: &'tcx Expr<'_>) -> bool {
        !ex.is_syntactic_place_expr() && self.has_sig_drop_attr(self.cx.typeck_results().expr_ty(ex))
    }

    fn has_sig_drop_attr(&mut self, ty: Ty<'tcx>) -> bool {
        self.seen_types.clear();
        self.has_sig_drop_attr_impl(ty)
    }

    fn has_sig_drop_attr_impl(&mut self, ty: Ty<'tcx>) -> bool {
        if let Some(adt) = ty.ty_adt_def()
            && get_attr(
                self.cx.sess(),
                self.cx.tcx.get_attrs_unchecked(adt.did()),
                "has_significant_drop",
            )
            .count()
                > 0
        {
            return true;
        }

        if !self.seen_types.insert(ty) {
            return false;
        }

        match ty.kind() {
            rustc_middle::ty::Adt(adt, args) => {
                // if some field has significant drop,
                adt.all_fields()
                    .map(|field| field.ty(self.cx.tcx, args))
                    .any(|ty| self.has_sig_drop_attr_impl(ty))
                    // or if there is no generic lifetime and..
                    // (to avoid false positive on `Ref<'a, MutexGuard<Foo>>`)
                    || (args
                        .iter()
                        .all(|arg| !matches!(arg.unpack(), GenericArgKind::Lifetime(_)))
                        // some generic parameter has significant drop
                        // (to avoid false negative on `Box<MutexGuard<Foo>>`)
                        && args
                            .iter()
                            .filter_map(|arg| match arg.unpack() {
                                GenericArgKind::Type(ty) => Some(ty),
                                _ => None,
                            })
                            .any(|ty| self.has_sig_drop_attr_impl(ty)))
            },
            rustc_middle::ty::Tuple(tys) => tys.iter().any(|ty| self.has_sig_drop_attr_impl(ty)),
            rustc_middle::ty::Array(ty, _) | rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr_impl(*ty),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum SigDropHolder {
    /// No values with significant drop present in this expression.
    ///
    /// Expressions that we've emitted lints do not count.
    None,
    /// Some field in this expression references to values with significant drop.
    ///
    /// Example: `(1, &data.lock().field)`.
    PackedRef,
    /// The value of this expression references to values with significant drop.
    ///
    /// Example: `data.lock().field`.
    DirectRef,
    /// This expression should be moved out to avoid significant drop in scrutinee.
    Moved,
}

impl Default for SigDropHolder {
    fn default() -> Self {
        Self::None
    }
}

struct SigDropHelper<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    parent_expr: Option<&'tcx Expr<'tcx>>,
    sig_drop_holder: SigDropHolder,
    sig_drop_spans: Vec<FoundSigDrop>,
    sig_drop_checker: SigDropChecker<'a, 'tcx>,
}

#[derive(Clone, Copy, Debug)]
struct FoundSigDrop {
    found_span: Span,
    is_unit_return_val: bool,
    peel_ref_times: usize,
}

impl<'a, 'tcx> SigDropHelper<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> SigDropHelper<'a, 'tcx> {
        SigDropHelper {
            cx,
            parent_expr: None,
            sig_drop_holder: SigDropHolder::None,
            sig_drop_spans: Vec::new(),
            sig_drop_checker: SigDropChecker::new(cx),
        }
    }

    fn find_sig_drop(&mut self, match_expr: &'tcx Expr<'_>) -> Vec<FoundSigDrop> {
        self.visit_expr(match_expr);

        core::mem::take(&mut self.sig_drop_spans)
    }

    fn replace_current_sig_drop(&mut self, found_span: Span, is_unit_return_val: bool, peel_ref_times: usize) {
        self.sig_drop_spans.clear();
        self.sig_drop_spans.push(FoundSigDrop {
            found_span,
            is_unit_return_val,
            peel_ref_times,
        });
    }

    fn try_move_sig_drop(&mut self, expr: &'tcx Expr<'_>, parent_expr: &'tcx Expr<'_>) {
        if self.sig_drop_holder == SigDropHolder::Moved {
            self.sig_drop_holder = SigDropHolder::None;
        }

        if self.sig_drop_holder == SigDropHolder::DirectRef {
            self.sig_drop_holder = SigDropHolder::PackedRef;
            self.try_move_sig_drop_direct_ref(expr, parent_expr);
        } else if self.sig_drop_checker.is_sig_drop_expr(expr) {
            // The values with significant drop can be moved to some other functions. For example, consider
            // `drop(data.lock())`. We use `SigDropHolder::None` here to avoid emitting lints in such scenarios.
            self.sig_drop_holder = SigDropHolder::None;
            self.try_move_sig_drop_direct_ref(expr, parent_expr);
        }

        if self.sig_drop_holder != SigDropHolder::None {
            let parent_ty = self.cx.typeck_results().expr_ty(parent_expr);
            if !ty_has_erased_regions(parent_ty) && !parent_expr.is_syntactic_place_expr() {
                self.replace_current_sig_drop(parent_expr.span, parent_ty.is_unit(), 0);
                self.sig_drop_holder = SigDropHolder::Moved;
            }

            let (peel_ref_ty, peel_ref_times) = ty_peel_refs(parent_ty);
            if !ty_has_erased_regions(peel_ref_ty) && is_copy(self.cx, peel_ref_ty) {
                self.replace_current_sig_drop(parent_expr.span, peel_ref_ty.is_unit(), peel_ref_times);
                self.sig_drop_holder = SigDropHolder::Moved;
            }
        }
    }

    fn try_move_sig_drop_direct_ref(&mut self, expr: &'tcx Expr<'_>, parent_expr: &'tcx Expr<'_>) {
        let arg_idx = match parent_expr.kind {
            ExprKind::MethodCall(_, receiver, exprs, _) => std::iter::once(receiver)
                .chain(exprs.iter())
                .find_position(|ex| ex.hir_id == expr.hir_id)
                .map(|(idx, _)| idx),
            ExprKind::Call(_, exprs) => exprs
                .iter()
                .find_position(|ex| ex.hir_id == expr.hir_id)
                .map(|(idx, _)| idx),
            ExprKind::Binary(_, lhs, rhs) | ExprKind::AssignOp(_, lhs, rhs) => [lhs, rhs]
                .iter()
                .find_position(|ex| ex.hir_id == expr.hir_id)
                .map(|(idx, _)| idx),
            ExprKind::Unary(_, ex) => (ex.hir_id == expr.hir_id).then_some(0),
            _ => {
                // Here we assume that all other expressions create or propagate the reference to the value with
                // significant drop.
                self.sig_drop_holder = SigDropHolder::DirectRef;
                return;
            },
        };
        let Some(arg_idx) = arg_idx else {
            return;
        };

        let fn_sig = if let Some(def_id) = self.cx.typeck_results().type_dependent_def_id(parent_expr.hir_id) {
            self.cx.tcx.fn_sig(def_id).instantiate_identity()
        } else {
            return;
        };

        let input_re = if let Some(input_ty) = fn_sig.skip_binder().inputs().get(arg_idx)
            && let rustc_middle::ty::Ref(input_re, _, _) = input_ty.kind()
        {
            input_re
        } else {
            return;
        };

        // Late bound lifetime parameters are not related to any constraints, so we can track them in a very
        // simple manner. For other lifetime parameters, we give up and update the state to `PackedRef`.
        let RegionKind::ReBound(_, input_re_bound) = input_re.kind() else {
            self.sig_drop_holder = SigDropHolder::PackedRef;
            return;
        };
        let contains_input_re = |re_bound| {
            if re_bound == input_re_bound {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };

        let output_ty = fn_sig.skip_binder().output();
        if let rustc_middle::ty::Ref(output_re, peel_ref_ty, _) = output_ty.kind()
            && input_re == output_re
            && for_each_top_level_late_bound_region(*peel_ref_ty, contains_input_re).is_continue()
        {
            // We're lucky! The output type is still a direct reference to the value with significant drop.
            self.sig_drop_holder = SigDropHolder::DirectRef;
        } else if for_each_top_level_late_bound_region(output_ty, contains_input_re).is_continue() {
            // The lifetime to the value with significant drop goes away. So we can emit a lint that suggests to
            // move the expression out.
            self.replace_current_sig_drop(parent_expr.span, output_ty.is_unit(), 0);
            self.sig_drop_holder = SigDropHolder::Moved;
        } else {
            // TODO: The lifetime is still there but it's for a inner type. For instance, consider
            // `Some(&mutex.lock().field)`, which has a type of `Option<&u32>`. How to address this scenario?
            self.sig_drop_holder = SigDropHolder::PackedRef;
        }
    }
}

fn ty_peel_refs(mut ty: Ty<'_>) -> (Ty<'_>, usize) {
    let mut n = 0;
    while let rustc_middle::ty::Ref(_, new_ty, Mutability::Not) = ty.kind() {
        ty = *new_ty;
        n += 1;
    }
    (ty, n)
}

fn ty_has_erased_regions(ty: Ty<'_>) -> bool {
    struct V;

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for V {
        type Result = ControlFlow<()>;

        fn visit_region(&mut self, region: Region<'tcx>) -> Self::Result {
            if region.is_erased() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    ty.visit_with(&mut V).is_break()
}

impl<'tcx> Visitor<'tcx> for SigDropHelper<'_, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        // We've emitted a lint on some neighborhood expression. That lint will suggest to move out the
        // _parent_ expression (not the expression itself). Since we decide to move out the parent
        // expression, it is pointless to continue to process the current expression.
        if self.sig_drop_holder == SigDropHolder::Moved {
            return;
        }

        // These states are of neighborhood expressions. We save and clear them here, and we'll later merge
        // the states of the current expression with them at the end of the method.
        let sig_drop_holder_before = core::mem::take(&mut self.sig_drop_holder);
        let sig_drop_spans_before = core::mem::take(&mut self.sig_drop_spans);
        let parent_expr_before = self.parent_expr.replace(ex);

        match ex.kind {
            // Skip blocks because values in blocks will be dropped as usual, and await
            // desugaring because temporary insides the future will have been dropped.
            ExprKind::Block(..) | ExprKind::Match(_, _, MatchSource::AwaitDesugar) => (),
            _ => walk_expr(self, ex),
        }

        if let Some(parent_ex) = parent_expr_before {
            match parent_ex.kind {
                ExprKind::Assign(lhs, _, _) | ExprKind::AssignOp(_, lhs, _)
                    if lhs.hir_id == ex.hir_id && self.sig_drop_holder == SigDropHolder::Moved =>
                {
                    // Never move out only the assignee. Instead, we should always move out the whole assignment.
                    self.replace_current_sig_drop(parent_ex.span, true, 0);
                },
                _ => {
                    self.try_move_sig_drop(ex, parent_ex);
                },
            }
        }

        self.sig_drop_holder = std::cmp::max(self.sig_drop_holder, sig_drop_holder_before);

        // We do not need those old spans in neighborhood expressions if we emit a lint that suggests to
        // move out the _parent_ expression (i.e., `self.sig_drop_holder == SigDropHolder::Moved`).
        if self.sig_drop_holder != SigDropHolder::Moved {
            let mut sig_drop_spans = sig_drop_spans_before;
            sig_drop_spans.append(&mut self.sig_drop_spans);
            self.sig_drop_spans = sig_drop_spans;
        }

        self.parent_expr = parent_expr_before;
    }
}

struct ArmSigDropHelper<'a, 'tcx> {
    sig_drop_checker: SigDropChecker<'a, 'tcx>,
    found_sig_drop_spans: FxIndexSet<Span>,
}

impl<'a, 'tcx> ArmSigDropHelper<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> ArmSigDropHelper<'a, 'tcx> {
        ArmSigDropHelper {
            sig_drop_checker: SigDropChecker::new(cx),
            found_sig_drop_spans: FxIndexSet::<Span>::default(),
        }
    }
}

fn has_significant_drop_in_arms<'tcx>(cx: &LateContext<'tcx>, arms: &[&'tcx Expr<'_>]) -> FxIndexSet<Span> {
    let mut helper = ArmSigDropHelper::new(cx);
    for arm in arms {
        helper.visit_expr(arm);
    }
    helper.found_sig_drop_spans
}

impl<'tcx> Visitor<'tcx> for ArmSigDropHelper<'_, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if self.sig_drop_checker.is_sig_drop_expr(ex) {
            self.found_sig_drop_spans.insert(ex.span);
            return;
        }
        walk_expr(self, ex);
    }
}
