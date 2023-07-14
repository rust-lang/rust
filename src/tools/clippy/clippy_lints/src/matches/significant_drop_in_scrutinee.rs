use crate::FxHashSet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{indent_of, snippet};
use clippy_utils::{get_attr, is_lint_allowed};
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::GenericArgKind;
use rustc_middle::ty::{Ty, TypeAndMut};
use rustc_span::Span;

use super::SIGNIFICANT_DROP_IN_SCRUTINEE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'_>],
    source: MatchSource,
) {
    if is_lint_allowed(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, expr.hir_id) {
        return;
    }

    if let Some((suggestions, message)) = has_significant_drop_in_scrutinee(cx, scrutinee, source) {
        for found in suggestions {
            span_lint_and_then(cx, SIGNIFICANT_DROP_IN_SCRUTINEE, found.found_span, message, |diag| {
                set_diagnostic(diag, cx, expr, found);
                let s = Span::new(expr.span.hi(), expr.span.hi(), expr.span.ctxt(), None);
                diag.span_label(s, "temporary lives until here");
                for span in has_significant_drop_in_arms(cx, arms) {
                    diag.span_label(span, "another value with significant `Drop` created here");
                }
                diag.note("this might lead to deadlocks or other unexpected behavior");
            });
        }
    }
}

fn set_diagnostic<'tcx>(diag: &mut Diagnostic, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, found: FoundSigDrop) {
    if found.lint_suggestion == LintSuggestion::MoveAndClone {
        // If our suggestion is to move and clone, then we want to leave it to the user to
        // decide how to address this lint, since it may be that cloning is inappropriate.
        // Therefore, we won't to emit a suggestion.
        return;
    }

    let original = snippet(cx, found.found_span, "..");
    let trailing_indent = " ".repeat(indent_of(cx, found.found_span).unwrap_or(0));

    let replacement = if found.lint_suggestion == LintSuggestion::MoveAndDerefToCopy {
        format!("let value = *{original};\n{trailing_indent}")
    } else if found.is_unit_return_val {
        // If the return value of the expression to be moved is unit, then we don't need to
        // capture the result in a temporary -- we can just replace it completely with `()`.
        format!("{original};\n{trailing_indent}")
    } else {
        format!("let value = {original};\n{trailing_indent}")
    };

    let suggestion_message = if found.lint_suggestion == LintSuggestion::MoveOnly {
        "try moving the temporary above the match"
    } else {
        "try moving the temporary above the match and create a copy"
    };

    let scrutinee_replacement = if found.is_unit_return_val {
        "()".to_owned()
    } else {
        "value".to_owned()
    };

    diag.multipart_suggestion(
        suggestion_message,
        vec![
            (expr.span.shrink_to_lo(), replacement),
            (found.found_span, scrutinee_replacement),
        ],
        Applicability::MaybeIncorrect,
    );
}

/// If the expression is an `ExprKind::Match`, check if the scrutinee has a significant drop that
/// may have a surprising lifetime.
fn has_significant_drop_in_scrutinee<'tcx>(
    cx: &LateContext<'tcx>,
    scrutinee: &'tcx Expr<'tcx>,
    source: MatchSource,
) -> Option<(Vec<FoundSigDrop>, &'static str)> {
    let mut helper = SigDropHelper::new(cx);
    let scrutinee = match (source, &scrutinee.kind) {
        (MatchSource::ForLoopDesugar, ExprKind::Call(_, [e])) => e,
        _ => scrutinee,
    };
    helper.find_sig_drop(scrutinee).map(|drops| {
        let message = if source == MatchSource::Normal {
            "temporary with significant `Drop` in `match` scrutinee will live until the end of the `match` expression"
        } else {
            "temporary with significant `Drop` in `for` loop condition will live until the end of the `for` expression"
        };
        (drops, message)
    })
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

    fn get_type(&self, ex: &'tcx Expr<'_>) -> Ty<'tcx> {
        self.cx.typeck_results().expr_ty(ex)
    }

    fn has_seen_type(&mut self, ty: Ty<'tcx>) -> bool {
        !self.seen_types.insert(ty)
    }

    fn has_sig_drop_attr(&mut self, cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
        if let Some(adt) = ty.ty_adt_def() {
            if get_attr(cx.sess(), cx.tcx.get_attrs_unchecked(adt.did()), "has_significant_drop").count() > 0 {
                return true;
            }
        }

        match ty.kind() {
            rustc_middle::ty::Adt(a, b) => {
                for f in a.all_fields() {
                    let ty = f.ty(cx.tcx, b);
                    if !self.has_seen_type(ty) && self.has_sig_drop_attr(cx, ty) {
                        return true;
                    }
                }

                for generic_arg in *b {
                    if let GenericArgKind::Type(ty) = generic_arg.unpack() {
                        if self.has_sig_drop_attr(cx, ty) {
                            return true;
                        }
                    }
                }
                false
            },
            rustc_middle::ty::Array(ty, _)
            | rustc_middle::ty::RawPtr(TypeAndMut { ty, .. })
            | rustc_middle::ty::Ref(_, ty, _)
            | rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr(cx, *ty),
            _ => false,
        }
    }
}

struct SigDropHelper<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    is_chain_end: bool,
    has_significant_drop: bool,
    current_sig_drop: Option<FoundSigDrop>,
    sig_drop_spans: Option<Vec<FoundSigDrop>>,
    special_handling_for_binary_op: bool,
    sig_drop_checker: SigDropChecker<'a, 'tcx>,
}

#[expect(clippy::enum_variant_names)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LintSuggestion {
    MoveOnly,
    MoveAndDerefToCopy,
    MoveAndClone,
}

#[derive(Clone, Copy)]
struct FoundSigDrop {
    found_span: Span,
    is_unit_return_val: bool,
    lint_suggestion: LintSuggestion,
}

impl<'a, 'tcx> SigDropHelper<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> SigDropHelper<'a, 'tcx> {
        SigDropHelper {
            cx,
            is_chain_end: true,
            has_significant_drop: false,
            current_sig_drop: None,
            sig_drop_spans: None,
            special_handling_for_binary_op: false,
            sig_drop_checker: SigDropChecker::new(cx),
        }
    }

    fn find_sig_drop(&mut self, match_expr: &'tcx Expr<'_>) -> Option<Vec<FoundSigDrop>> {
        self.visit_expr(match_expr);

        // If sig drop spans is empty but we found a significant drop, it means that we didn't find
        // a type that was trivially copyable as we moved up the chain after finding a significant
        // drop, so move the entire scrutinee.
        if self.has_significant_drop && self.sig_drop_spans.is_none() {
            self.try_setting_current_suggestion(match_expr, true);
            self.move_current_suggestion();
        }

        self.sig_drop_spans.take()
    }

    fn replace_current_sig_drop(
        &mut self,
        found_span: Span,
        is_unit_return_val: bool,
        lint_suggestion: LintSuggestion,
    ) {
        self.current_sig_drop.replace(FoundSigDrop {
            found_span,
            is_unit_return_val,
            lint_suggestion,
        });
    }

    /// This will try to set the current suggestion (so it can be moved into the suggestions vec
    /// later). If `allow_move_and_clone` is false, the suggestion *won't* be set -- this gives us
    /// an opportunity to look for another type in the chain that will be trivially copyable.
    /// However, if we are at the end of the chain, we want to accept whatever is there. (The
    /// suggestion won't actually be output, but the diagnostic message will be output, so the user
    /// can determine the best way to handle the lint.)
    fn try_setting_current_suggestion(&mut self, expr: &'tcx Expr<'_>, allow_move_and_clone: bool) {
        if self.current_sig_drop.is_some() {
            return;
        }
        let ty = self.sig_drop_checker.get_type(expr);
        if ty.is_ref() {
            // We checked that the type was ref, so builtin_deref will return Some TypeAndMut,
            // but let's avoid any chance of an ICE
            if let Some(TypeAndMut { ty, .. }) = ty.builtin_deref(true) {
                if ty.is_trivially_pure_clone_copy() {
                    self.replace_current_sig_drop(expr.span, false, LintSuggestion::MoveAndDerefToCopy);
                } else if allow_move_and_clone {
                    self.replace_current_sig_drop(expr.span, false, LintSuggestion::MoveAndClone);
                }
            }
        } else if ty.is_trivially_pure_clone_copy() {
            self.replace_current_sig_drop(expr.span, false, LintSuggestion::MoveOnly);
        } else if allow_move_and_clone {
            self.replace_current_sig_drop(expr.span, false, LintSuggestion::MoveAndClone);
        }
    }

    fn move_current_suggestion(&mut self) {
        if let Some(current) = self.current_sig_drop.take() {
            self.sig_drop_spans.get_or_insert_with(Vec::new).push(current);
        }
    }

    fn visit_exprs_for_binary_ops(
        &mut self,
        left: &'tcx Expr<'_>,
        right: &'tcx Expr<'_>,
        is_unit_return_val: bool,
        span: Span,
    ) {
        self.special_handling_for_binary_op = true;
        self.visit_expr(left);
        self.visit_expr(right);

        // If either side had a significant drop, suggest moving the entire scrutinee to avoid
        // unnecessary copies and to simplify cases where both sides have significant drops.
        if self.has_significant_drop {
            self.replace_current_sig_drop(span, is_unit_return_val, LintSuggestion::MoveOnly);
        }

        self.special_handling_for_binary_op = false;
    }
}

impl<'a, 'tcx> Visitor<'tcx> for SigDropHelper<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        if !self.is_chain_end
            && self
                .sig_drop_checker
                .has_sig_drop_attr(self.cx, self.sig_drop_checker.get_type(ex))
        {
            self.has_significant_drop = true;
            return;
        }
        self.is_chain_end = false;

        match ex.kind {
            ExprKind::MethodCall(_, expr, ..) => {
                self.visit_expr(expr);
            }
            ExprKind::Binary(_, left, right) => {
                self.visit_exprs_for_binary_ops(left, right, false, ex.span);
            }
            ExprKind::Assign(left, right, _) | ExprKind::AssignOp(_, left, right) => {
                self.visit_exprs_for_binary_ops(left, right, true, ex.span);
            }
            ExprKind::Tup(exprs) => {
                for expr in exprs {
                    self.visit_expr(expr);
                    if self.has_significant_drop {
                        // We may have not have set current_sig_drop if all the suggestions were
                        // MoveAndClone, so add this tuple item's full expression in that case.
                        if self.current_sig_drop.is_none() {
                            self.try_setting_current_suggestion(expr, true);
                        }

                        // Now we are guaranteed to have something, so add it to the final vec.
                        self.move_current_suggestion();
                    }
                    // Reset `has_significant_drop` after each tuple expression so we can look for
                    // additional cases.
                    self.has_significant_drop = false;
                }
                if self.sig_drop_spans.is_some() {
                    self.has_significant_drop = true;
                }
            }
            ExprKind::Array(..) |
            ExprKind::Call(..) |
            ExprKind::Unary(..) |
            ExprKind::If(..) |
            ExprKind::Match(..) |
            ExprKind::Field(..) |
            ExprKind::Index(..) |
            ExprKind::Ret(..) |
            ExprKind::Become(..) |
            ExprKind::Repeat(..) |
            ExprKind::Yield(..) => walk_expr(self, ex),
            ExprKind::AddrOf(_, _, _) |
            ExprKind::Block(_, _) |
            ExprKind::Break(_, _) |
            ExprKind::Cast(_, _) |
            // Don't want to check the closure itself, only invocation, which is covered by MethodCall
            ExprKind::Closure { .. } |
            ExprKind::ConstBlock(_) |
            ExprKind::Continue(_) |
            ExprKind::DropTemps(_) |
            ExprKind::Err(_) |
            ExprKind::InlineAsm(_) |
            ExprKind::OffsetOf(_, _) |
            ExprKind::Let(_) |
            ExprKind::Lit(_) |
            ExprKind::Loop(_, _, _, _) |
            ExprKind::Path(_) |
            ExprKind::Struct(_, _, _) |
            ExprKind::Type(_, _) => {
                return;
            }
        }

        // Once a significant temporary has been found, we need to go back up at least 1 level to
        // find the span to extract for replacement, so the temporary gets dropped. However, for
        // binary ops, we want to move the whole scrutinee so we avoid unnecessary copies and to
        // simplify cases where both sides have significant drops.
        if self.has_significant_drop && !self.special_handling_for_binary_op {
            self.try_setting_current_suggestion(ex, false);
        }
    }
}

struct ArmSigDropHelper<'a, 'tcx> {
    sig_drop_checker: SigDropChecker<'a, 'tcx>,
    found_sig_drop_spans: FxHashSet<Span>,
}

impl<'a, 'tcx> ArmSigDropHelper<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> ArmSigDropHelper<'a, 'tcx> {
        ArmSigDropHelper {
            sig_drop_checker: SigDropChecker::new(cx),
            found_sig_drop_spans: FxHashSet::<Span>::default(),
        }
    }
}

fn has_significant_drop_in_arms<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'_>]) -> FxHashSet<Span> {
    let mut helper = ArmSigDropHelper::new(cx);
    for arm in arms {
        helper.visit_expr(arm.body);
    }
    helper.found_sig_drop_spans
}

impl<'a, 'tcx> Visitor<'tcx> for ArmSigDropHelper<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if self
            .sig_drop_checker
            .has_sig_drop_attr(self.sig_drop_checker.cx, self.sig_drop_checker.get_type(ex))
        {
            self.found_sig_drop_spans.insert(ex.span);
            return;
        }
        walk_expr(self, ex);
    }
}
