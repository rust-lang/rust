use std::ops::ControlFlow;

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::sugg::Sugg;
use clippy_utils::visitors::is_const_evaluatable;
use clippy_utils::{is_in_const_context, is_mutable, sym};
use rustc_ast::Mutability;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId, LangItem};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::adjustment::{Adjust, DerefAdjustKind, OverloadedDeref};
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;

use crate::methods::is_clone_like;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for slice references with cloned references such as `&[f.clone()]`.
    ///
    /// ### Why is this bad
    ///
    /// A reference does not need to be owned in order to be used as a slice.
    ///
    /// ### Known problems
    ///
    /// This lint does not know whether or not a clone implementation has side effects.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// let data = 10;
    /// let data_ref = &data;
    /// take_slice(&[data_ref.clone()]);
    /// ```
    /// Use instead:
    /// ```ignore
    /// use std::slice;
    /// let data = 10;
    /// let data_ref = &data;
    /// take_slice(slice::from_ref(data_ref));
    /// ```
    #[clippy::version = "1.89.0"]
    pub CLONED_REF_TO_SLICE_REFS,
    perf,
    "cloning a reference for slice references"
}

impl_lint_pass!(ClonedRefToSliceRefs<'_> => [CLONED_REF_TO_SLICE_REFS]);

pub struct ClonedRefToSliceRefs<'a> {
    msrv: &'a Msrv,
}
impl<'a> ClonedRefToSliceRefs<'a> {
    pub fn new(conf: &'a Conf) -> Self {
        Self { msrv: &conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for ClonedRefToSliceRefs<'_> {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if self.msrv.meets(cx, {
            if is_in_const_context(cx) {
                msrvs::CONST_SLICE_FROM_REF
            } else {
                msrvs::SLICE_FROM_REF
            }
        })
            // `&[foo.clone()]` expressions
            && let ExprKind::AddrOf(_, mutability, arr) = &expr.kind
            // mutable references would have a different meaning
            && mutability.is_not()

            // check for single item arrays
            && let ExprKind::Array([item]) = &arr.kind

            // check for clones
            && let ExprKind::MethodCall(path, recv, _, _) = item.kind
            && let Some(adjustment) = is_needless_clone_or_equivalent(cx, recv, path.ident.name, item.hir_id)

            // check for immutability or purity
            && (!is_mutable(cx, recv) || is_const_evaluatable(cx, recv))

            // get appropriate crate for `slice::from_ref`
            && let Some(builtin_crate) = clippy_utils::std_or_core(cx)
        {
            let mut applicability = Applicability::MachineApplicable;
            let sugg = Sugg::hir_with_context(cx, recv, expr.span.ctxt(), "_", &mut applicability);

            span_lint_and_sugg(
                cx,
                CLONED_REF_TO_SLICE_REFS,
                expr.span,
                format!(
                    "unnecessary use of `{}` to create a slice from a reference",
                    path.ident.name
                ),
                "try",
                format!("{builtin_crate}::slice::from_ref({adjustment}{sugg})"),
                applicability,
            );
        }
    }
}

/// Checks if a method call is a needless clone or equivalent. If so, returns the necessary
/// adjustments to use the method receiver directly without cloning.
/// For example, in the code below:
/// ```rust,no_run
/// use std::path::PathBuf;
///
/// let w = &PathBuf::new();
/// let b = &[w.to_path_buf()];
/// ```
/// We would replace `&[w.to_path_buf()]` with `std::slice::from_ref(&*w)`,
/// hence we return `Some("&*")` as the adjustment.
fn is_needless_clone_or_equivalent<'tcx>(
    cx: &LateContext<'tcx>,
    method_recv: &'tcx Expr<'tcx>,
    method_name: Symbol,
    hir_id: HirId,
) -> Option<String> {
    let method_def = cx.ty_based_def(hir_id).opt_parent(cx)?;
    if !method_def.is_lang_item(cx, LangItem::Clone) && !is_clone_like(cx, method_name, method_def) {
        return None;
    }

    let method_ret_ty = cx.typeck_results().node_type(hir_id);
    let method_recv_ty = cx.typeck_results().expr_ty_adjusted(method_recv);
    let ty::Ref(_, method_recv_ty_inner, Mutability::Not) = method_recv_ty.kind() else {
        return None;
    };

    let method_recv_adjustments = cx.typeck_results().expr_adjustments(method_recv);

    // The return type of the clone-like method should be the same as the inner type of the reference
    // being cloned, except for the following special cases:
    // 1. `OsString`, which is first dereferenced to `OsStr` and the borrowed as `&OsStr`.
    // 2. `PathBuf`, which is first dereferenced to `Path` and then borrowed as `&Path`.
    let adjust_target_ty = if method_ret_ty == *method_recv_ty_inner {
        method_ret_ty
    } else if let Some(after_special_case_ty_name @ (sym::OsStr | sym::Path)) = method_recv_ty_inner.opt_diag_name(cx)
        // Looking for the `OSString -> OSStr` or `PathBuf -> Path` adjustment in the abovementioned special cases
        && let [preceeding_derefs @ .., special_case, last_borrow] = method_recv_adjustments
        && matches!(
            special_case.kind,
            Adjust::Deref(DerefAdjustKind::Overloaded(OverloadedDeref {
                mutbl: Mutability::Not,
                ..
            }))
        )
        && matches!(last_borrow.kind, Adjust::Borrow(_))
        && special_case.target.is_diag_item(cx, after_special_case_ty_name)
        && let before_special_case_ty = preceeding_derefs
            .last().map_or_else(|| cx.typeck_results().expr_ty(method_recv), |a| a.target)
        && matches!(
            (before_special_case_ty.opt_diag_name(cx)?, after_special_case_ty_name),
            (sym::OsString, sym::OsStr) | (sym::PathBuf, sym::Path))
    {
        before_special_case_ty
    } else {
        return None;
    };

    // Find the number of adjustments required until `method_recv_ty_source` becomes `adjust_target_ty`
    let method_recv_ty_source = cx.typeck_results().expr_ty(method_recv);
    let adjust_count = method_recv_adjustments
        .iter()
        .enumerate()
        .try_fold(method_recv_ty_source, |ty, (i, a)| {
            if ty == adjust_target_ty {
                ControlFlow::Break(i)
            } else {
                ControlFlow::Continue(a.target)
            }
        })
        .break_value()?;

    let (needs_borrow, deref_count) = if adjust_count == 0 || !method_recv_ty_source.is_ref() {
        (true, adjust_count)
    } else {
        (false, adjust_count - 1)
    };

    Some(if needs_borrow {
        format!("&{}", "*".repeat(deref_count))
    } else {
        "*".repeat(deref_count)
    })
}
