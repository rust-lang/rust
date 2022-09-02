use crate::{LateContext, LateLintPass, LintContext};
use rustc_errors::{fluent, Applicability};
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::symbol::sym;
use rustc_span::Span;

declare_lint! {
    /// The `array_into_iter` lint detects calling `into_iter` on arrays.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
    /// # #![allow(unused)]
    /// [1, 2, 3].into_iter().for_each(|n| { *n; });
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Since Rust 1.53, arrays implement `IntoIterator`. However, to avoid
    /// breakage, `array.into_iter()` in Rust 2015 and 2018 code will still
    /// behave as `(&array).into_iter()`, returning an iterator over
    /// references, just like in Rust 1.52 and earlier.
    /// This only applies to the method call syntax `array.into_iter()`, not to
    /// any other syntax such as `for _ in array` or `IntoIterator::into_iter(array)`.
    pub ARRAY_INTO_ITER,
    Warn,
    "detects calling `into_iter` on arrays in Rust 2015 and 2018",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html>",
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2021),
    };
}

#[derive(Copy, Clone, Default)]
pub struct ArrayIntoIter {
    for_expr_span: Span,
}

impl_lint_pass!(ArrayIntoIter => [ARRAY_INTO_ITER]);

impl<'tcx> LateLintPass<'tcx> for ArrayIntoIter {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        // Save the span of expressions in `for _ in expr` syntax,
        // so we can give a better suggestion for those later.
        if let hir::ExprKind::Match(arg, [_], hir::MatchSource::ForLoopDesugar) = &expr.kind {
            if let hir::ExprKind::Call(path, [arg]) = &arg.kind {
                if let hir::ExprKind::Path(hir::QPath::LangItem(
                    hir::LangItem::IntoIterIntoIter,
                    ..,
                )) = &path.kind
                {
                    self.for_expr_span = arg.span;
                }
            }
        }

        // We only care about method call expressions.
        if let hir::ExprKind::MethodCall(call, receiver_arg, ..) = &expr.kind {
            if call.ident.name != sym::into_iter {
                return;
            }

            // Check if the method call actually calls the libcore
            // `IntoIterator::into_iter`.
            let def_id = cx.typeck_results().type_dependent_def_id(expr.hir_id).unwrap();
            match cx.tcx.trait_of_item(def_id) {
                Some(trait_id) if cx.tcx.is_diagnostic_item(sym::IntoIterator, trait_id) => {}
                _ => return,
            };

            // As this is a method call expression, we have at least one argument.
            let receiver_ty = cx.typeck_results().expr_ty(receiver_arg);
            let adjustments = cx.typeck_results().expr_adjustments(receiver_arg);

            let Some(Adjustment { kind: Adjust::Borrow(_), target }) = adjustments.last() else {
                return
            };

            let types =
                std::iter::once(receiver_ty).chain(adjustments.iter().map(|adj| adj.target));

            let mut found_array = false;

            for ty in types {
                match ty.kind() {
                    // If we run into a &[T; N] or &[T] first, there's nothing to warn about.
                    // It'll resolve to the reference version.
                    ty::Ref(_, inner_ty, _) if inner_ty.is_array() => return,
                    ty::Ref(_, inner_ty, _) if matches!(inner_ty.kind(), ty::Slice(..)) => return,
                    // Found an actual array type without matching a &[T; N] first.
                    // This is the problematic case.
                    ty::Array(..) => {
                        found_array = true;
                        break;
                    }
                    _ => {}
                }
            }

            if !found_array {
                return;
            }

            // Emit lint diagnostic.
            let target = match *target.kind() {
                ty::Ref(_, inner_ty, _) if inner_ty.is_array() => "[T; N]",
                ty::Ref(_, inner_ty, _) if matches!(inner_ty.kind(), ty::Slice(..)) => "[T]",
                // We know the original first argument type is an array type,
                // we know that the first adjustment was an autoref coercion
                // and we know that `IntoIterator` is the trait involved. The
                // array cannot be coerced to something other than a reference
                // to an array or to a slice.
                _ => bug!("array type coerced to something other than array or slice"),
            };
            cx.struct_span_lint(ARRAY_INTO_ITER, call.ident.span, |lint| {
                let mut diag = lint.build(fluent::lint::array_into_iter);
                diag.set_arg("target", target);
                diag.span_suggestion(
                    call.ident.span,
                    fluent::lint::use_iter_suggestion,
                    "iter",
                    Applicability::MachineApplicable,
                );
                if self.for_expr_span == expr.span {
                    diag.span_suggestion(
                        receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
                        fluent::lint::remove_into_iter_suggestion,
                        "",
                        Applicability::MaybeIncorrect,
                    );
                } else if receiver_ty.is_array() {
                    diag.multipart_suggestion(
                        fluent::lint::use_explicit_into_iter_suggestion,
                        vec![
                            (expr.span.shrink_to_lo(), "IntoIterator::into_iter(".into()),
                            (
                                receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
                                ")".into(),
                            ),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                }
                diag.emit();
            })
        }
    }
}
