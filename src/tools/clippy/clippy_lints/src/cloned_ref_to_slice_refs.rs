use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::visitors::is_const_evaluatable;
use clippy_utils::{is_in_const_context, is_mutable, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for slice references with cloned references such as `&[f.clone()]`.
    ///
    /// ### Why is this bad
    ///
    /// A reference does not need to be owned in order to used as a slice.
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
    #[clippy::version = "1.87.0"]
    pub CLONED_REF_TO_SLICE_REFS,
    perf,
    "cloning a reference for slice references"
}

pub struct ClonedRefToSliceRefs<'a> {
    msrv: &'a Msrv,
}
impl<'a> ClonedRefToSliceRefs<'a> {
    pub fn new(conf: &'a Conf) -> Self {
        Self { msrv: &conf.msrv }
    }
}

impl_lint_pass!(ClonedRefToSliceRefs<'_> => [CLONED_REF_TO_SLICE_REFS]);

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
            && let ExprKind::MethodCall(_, val, _, _) = item.kind
            && is_trait_method(cx, item, sym::Clone)

            // check for immutability or purity
            && (!is_mutable(cx, val) || is_const_evaluatable(cx, val))

            // get appropriate crate for `slice::from_ref`
            && let Some(builtin_crate) = clippy_utils::std_or_core(cx)
        {
            let mut sugg = Sugg::hir(cx, val, "_");
            if !cx.typeck_results().expr_ty(val).is_ref() {
                sugg = sugg.addr();
            }

            span_lint_and_sugg(
                cx,
                CLONED_REF_TO_SLICE_REFS,
                expr.span,
                format!("this call to `clone` can be replaced with `{builtin_crate}::slice::from_ref`"),
                "try",
                format!("{builtin_crate}::slice::from_ref({sugg})"),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
