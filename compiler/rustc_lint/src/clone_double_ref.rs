use crate::lints::{CloneDoubleRefExplicit, CloneDoubleRefTryDeref};
use crate::{lints, LateLintPass, LintContext};

use rustc_hir as hir;
use rustc_hir::ExprKind;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::sym;

declare_lint! {
    /// The `clone_double_ref` lint checks for usage of `.clone()` on an `&&T`,
    /// which copies the inner `&T`, instead of cloning the underlying `T` and
    /// can be confusing.
    pub CLONE_DOUBLE_REF,
    Warn,
    "using `clone` on `&&T`"
}

declare_lint_pass!(CloneDoubleRef => [CLONE_DOUBLE_REF]);

impl<'tcx> LateLintPass<'tcx> for CloneDoubleRef {
    fn check_expr(&mut self, cx: &crate::LateContext<'tcx>, e: &'tcx hir::Expr<'tcx>) {
        let ExprKind::MethodCall(path, receiver, args, _) = &e.kind else { return; };

        if path.ident.name != sym::clone || !args.is_empty() {
            return;
        }

        let typeck_results = cx.typeck_results();

        if typeck_results
            .type_dependent_def_id(e.hir_id)
            .and_then(|id| cx.tcx.trait_of_item(id))
            .zip(cx.tcx.lang_items().clone_trait())
            .map_or(true, |(x, y)| x != y)
        {
            return;
        }

        let arg_adjustments = cx.typeck_results().expr_adjustments(receiver);

        // https://github.com/rust-lang/rust-clippy/issues/9272
        if arg_adjustments.iter().any(|adj| matches!(adj.kind, Adjust::Deref(Some(_)))) {
            return;
        }

        if !receiver.span.eq_ctxt(e.span) {
            return;
        }

        let arg_ty = arg_adjustments
            .last()
            .map_or_else(|| cx.typeck_results().expr_ty(receiver), |a| a.target);

        let ty = cx.typeck_results().expr_ty(e);

        if let ty::Ref(_, inner, _) = arg_ty.kind()
            && let ty::Ref(_, innermost, _) = inner.kind() {
                let mut inner_ty = innermost;
                let mut n = 1;
                while let ty::Ref(_, inner, _) = inner_ty.kind() {
                    inner_ty = inner;
                    n += 1;
                }
                let refs = "&".repeat(n);
                let derefs = "*".repeat(n);
                let start = e.span.with_hi(receiver.span.lo());
                let end = e.span.with_lo(receiver.span.hi());
                cx.emit_spanned_lint(CLONE_DOUBLE_REF, e.span, lints::CloneDoubleRef {
                    ty,
                    try_deref: CloneDoubleRefTryDeref {
                        start,
                        end,
                        refs: refs.clone(),
                        derefs,
                    },
                    explicit: CloneDoubleRefExplicit {
                        start,
                        end,
                        refs,
                        ty: *inner_ty,
                    }
                });
        }
    }
}
