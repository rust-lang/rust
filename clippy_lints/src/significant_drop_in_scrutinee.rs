use crate::FxHashSet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::get_attr;
use clippy_utils::source::{indent_of, snippet};
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{Ty, TypeAndMut};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Check for temporaries returned from function calls in a match scrutinee that have the
    /// `clippy::has_significant_drop` attribute.
    ///
    /// ### Why is this bad?
    /// The `clippy::has_significant_drop` attribute can be added to types whose Drop impls have
    /// an important side-effect, such as unlocking a mutex, making it important for users to be
    /// able to accurately understand their lifetimes. When a temporary is returned in a function
    /// call in a match scrutinee, its lifetime lasts until the end of the match block, which may
    /// be surprising.
    ///
    /// For `Mutex`es this can lead to a deadlock. This happens when the match scrutinee uses a
    /// function call that returns a `MutexGuard` and then tries to lock again in one of the match
    /// arms. In that case the `MutexGuard` in the scrutinee will not be dropped until the end of
    /// the match block and thus will not unlock.
    ///
    /// ### Example
    /// ```rust
    /// # use std::sync::Mutex;
    ///
    /// # struct State {}
    ///
    /// # impl State {
    /// #     fn foo(&self) -> bool {
    /// #         true
    /// #     }
    ///
    /// #     fn bar(&self) {}
    /// # }
    ///
    ///
    /// let mutex = Mutex::new(State {});
    ///
    /// match mutex.lock().unwrap().foo() {
    ///     true => {
    ///         mutex.lock().unwrap().bar(); // Deadlock!
    ///     }
    ///     false => {}
    /// };
    ///
    /// println!("All done!");
    ///
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::sync::Mutex;
    ///
    /// # struct State {}
    ///
    /// # impl State {
    /// #     fn foo(&self) -> bool {
    /// #         true
    /// #     }
    ///
    /// #     fn bar(&self) {}
    /// # }
    ///
    /// let mutex = Mutex::new(State {});
    ///
    /// let is_foo = mutex.lock().unwrap().foo();
    /// match is_foo {
    ///     true => {
    ///         mutex.lock().unwrap().bar();
    ///     }
    ///     false => {}
    /// };
    ///
    /// println!("All done!");
    /// ```
    #[clippy::version = "1.60.0"]
    pub SIGNIFICANT_DROP_IN_SCRUTINEE,
    nursery,
    "warns when a temporary of a type with a drop with a significant side-effect might have a surprising lifetime"
}

declare_lint_pass!(SignificantDropInScrutinee => [SIGNIFICANT_DROP_IN_SCRUTINEE]);

impl<'tcx> LateLintPass<'tcx> for SignificantDropInScrutinee {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(suggestions) = has_significant_drop_in_scrutinee(cx, expr) {
            for found in suggestions {
                span_lint_and_then(
                    cx,
                    SIGNIFICANT_DROP_IN_SCRUTINEE,
                    found.found_span,
                    "temporary with significant drop in match scrutinee",
                    |diag| set_diagnostic(diag, cx, expr, found),
                )
            }
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
        format!("let value = *{};\n{}", original, trailing_indent)
    } else if found.is_unit_return_val {
        // If the return value of the expression to be moved is unit, then we don't need to
        // capture the result in a temporary -- we can just replace it completely with `()`.
        format!("{};\n{}", original, trailing_indent)
    } else {
        format!("let value = {};\n{}", original, trailing_indent)
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

/// If the expression is an ExprKind::Match, check if the scrutinee has a significant drop that may
/// have a surprising lifetime.
fn has_significant_drop_in_scrutinee<'tcx, 'a>(
    cx: &'a LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<Vec<FoundSigDrop>> {
    let mut helper = SigDropHelper::new(cx);
    match expr.kind {
        ExprKind::Match(match_expr, _, _) => helper.find_sig_drop(match_expr),
        _ => None,
    }
}

struct SigDropHelper<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    is_chain_end: bool,
    seen_types: FxHashSet<Ty<'tcx>>,
    has_significant_drop: bool,
    current_sig_drop: Option<FoundSigDrop>,
    sig_drop_spans: Option<Vec<FoundSigDrop>>,
    special_handling_for_binary_op: bool,
}

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
            seen_types: FxHashSet::default(),
            has_significant_drop: false,
            current_sig_drop: None,
            sig_drop_spans: None,
            special_handling_for_binary_op: false,
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

    /// This will try to set the current suggestion (so it can be moved into the suggestions vec
    /// later). If allow_move_and_clone is false, the suggestion *won't* be set -- this gives us
    /// an opportunity to look for another type in the chain that will be trivially copyable.
    /// However, if we are at the the end of the chain, we want to accept whatever is there. (The
    /// suggestion won't actually be output, but the diagnostic message will be output, so the user
    /// can determine the best way to handle the lint.)
    fn try_setting_current_suggestion(&mut self, expr: &'tcx Expr<'_>, allow_move_and_clone: bool) {
        if self.current_sig_drop.is_some() {
            return;
        }
        let ty = self.get_type(expr);
        if ty.is_ref() {
            // We checked that the type was ref, so builtin_deref will return Some TypeAndMut,
            // but let's avoid any chance of an ICE
            if let Some(TypeAndMut { ty, .. }) = ty.builtin_deref(true) {
                if ty.is_trivially_pure_clone_copy() {
                    self.current_sig_drop.replace(FoundSigDrop {
                        found_span: expr.span,
                        is_unit_return_val: false,
                        lint_suggestion: LintSuggestion::MoveAndDerefToCopy,
                    });
                } else if allow_move_and_clone {
                    self.current_sig_drop.replace(FoundSigDrop {
                        found_span: expr.span,
                        is_unit_return_val: false,
                        lint_suggestion: LintSuggestion::MoveAndClone,
                    });
                }
            }
        } else if ty.is_trivially_pure_clone_copy() {
            self.current_sig_drop.replace(FoundSigDrop {
                found_span: expr.span,
                is_unit_return_val: false,
                lint_suggestion: LintSuggestion::MoveOnly,
            });
        }
    }

    fn move_current_suggestion(&mut self) {
        if let Some(current) = self.current_sig_drop.take() {
            self.sig_drop_spans.get_or_insert_with(Vec::new).push(current);
        }
    }

    fn get_type(&self, ex: &'tcx Expr<'_>) -> Ty<'tcx> {
        self.cx.typeck_results().expr_ty(ex)
    }

    fn has_seen_type(&mut self, ty: Ty<'tcx>) -> bool {
        !self.seen_types.insert(ty)
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
            self.current_sig_drop.replace(FoundSigDrop {
                found_span: span,
                is_unit_return_val,
                lint_suggestion: LintSuggestion::MoveOnly,
            });
        }

        self.special_handling_for_binary_op = false;
    }

    fn has_sig_drop_attr(&mut self, cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
        if let Some(adt) = ty.ty_adt_def() {
            if get_attr(cx.sess(), cx.tcx.get_attrs(adt.did()), "has_significant_drop").count() > 0 {
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

                for generic_arg in b.iter() {
                    if let GenericArgKind::Type(ty) = generic_arg.unpack() {
                        if self.has_sig_drop_attr(cx, ty) {
                            return true;
                        }
                    }
                }
                false
            },
            rustc_middle::ty::Array(ty, _) => self.has_sig_drop_attr(cx, *ty),
            rustc_middle::ty::RawPtr(TypeAndMut { ty, .. }) => self.has_sig_drop_attr(cx, *ty),
            rustc_middle::ty::Ref(_, ty, _) => self.has_sig_drop_attr(cx, *ty),
            rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr(cx, *ty),
            _ => false,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for SigDropHelper<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        if !self.is_chain_end && self.has_sig_drop_attr(self.cx, self.get_type(ex)) {
            self.has_significant_drop = true;
            return;
        }
        self.is_chain_end = false;

        match ex.kind {
            ExprKind::MethodCall(_, [ref expr, ..], _) => {
                self.visit_expr(expr)
            }
            ExprKind::Binary(_, left, right) => {
                self.visit_exprs_for_binary_ops(left, right, false, ex.span);
            }
            ExprKind::Assign(left, right, _) => {
                self.visit_exprs_for_binary_ops(left, right, true, ex.span);
            }
            ExprKind::AssignOp(_, left, right) => {
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
            ExprKind::Box(..) |
                ExprKind::Array(..) |
                ExprKind::Call(..) |
                ExprKind::Unary(..) |
                ExprKind::If(..) |
                ExprKind::Match(..) |
                ExprKind::Field(..) |
                ExprKind::Index(..) |
                ExprKind::Ret(..) |
                ExprKind::Repeat(..) |
                ExprKind::Yield(..) |
                ExprKind::MethodCall(..) => walk_expr(self, ex),
            ExprKind::AddrOf(_, _, _) |
                ExprKind::Block(_, _) |
                ExprKind::Break(_, _) |
                ExprKind::Cast(_, _) |
                // Don't want to check the closure itself, only invocation, which is covered by MethodCall
                ExprKind::Closure(_, _, _, _, _) |
                ExprKind::ConstBlock(_) |
                ExprKind::Continue(_) |
                ExprKind::DropTemps(_) |
                ExprKind::Err |
                ExprKind::InlineAsm(_) |
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
