use clippy_utils::diagnostics::span_lint_and_then;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, ConstKind};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Pos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for large `const` arrays that should
    /// be defined as `static` instead.
    ///
    /// ### Why is this bad?
    /// Performance: const variables are inlined upon use.
    /// Static items result in only one instance and has a fixed location in memory.
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub const a = [0u32; 1_000_000];
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// pub static a = [0u32; 1_000_000];
    /// ```
    #[clippy::version = "1.44.0"]
    pub LARGE_CONST_ARRAYS,
    perf,
    "large non-scalar const array may cause performance overhead"
}

pub struct LargeConstArrays {
    maximum_allowed_size: u128,
}

impl LargeConstArrays {
    #[must_use]
    pub fn new(maximum_allowed_size: u128) -> Self {
        Self { maximum_allowed_size }
    }
}

impl_lint_pass!(LargeConstArrays => [LARGE_CONST_ARRAYS]);

impl<'tcx> LateLintPass<'tcx> for LargeConstArrays {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if_chain! {
            if !item.span.from_expansion();
            if let ItemKind::Const(hir_ty, _) = &item.kind;
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if let ty::Array(element_type, cst) = ty.kind();
            if let ConstKind::Value(ty::ValTree::Leaf(element_count)) = cst.kind();
            if let Ok(element_count) = element_count.try_to_target_usize(cx.tcx);
            if let Ok(element_size) = cx.layout_of(*element_type).map(|l| l.size.bytes());
            if self.maximum_allowed_size < u128::from(element_count) * u128::from(element_size);

            then {
                let hi_pos = item.ident.span.lo() - BytePos::from_usize(1);
                let sugg_span = Span::new(
                    hi_pos - BytePos::from_usize("const".len()),
                    hi_pos,
                    item.span.ctxt(),
                    item.span.parent(),
                );
                span_lint_and_then(
                    cx,
                    LARGE_CONST_ARRAYS,
                    item.span,
                    "large array defined as const",
                    |diag| {
                        diag.span_suggestion(
                            sugg_span,
                            "make this a static item",
                            "static",
                            Applicability::MachineApplicable,
                        );
                    }
                );
            }
        }
    }
}
