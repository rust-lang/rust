use crate::rustc_target::abi::LayoutOf;
use crate::utils::span_lint_and_then;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty::{self, ConstKind};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Pos, Span};
use rustc_typeck::hir_ty_to_ty;

declare_clippy_lint! {
    /// **What it does:** Checks for large `const` arrays that should
    /// be defined as `static` instead.
    ///
    /// **Why is this bad?** Performance: const variables are inlined upon use.
    /// Static items result in only one instance and has a fixed location in memory.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// // Bad
    /// pub const a = [0u32; 1_000_000];
    ///
    /// // Good
    /// pub static a = [0u32; 1_000_000];
    /// ```
    pub LARGE_CONST_ARRAYS,
    perf,
    "large non-scalar const array may cause performance overhead"
}

pub struct LargeConstArrays {
    maximum_allowed_size: u64,
}

impl LargeConstArrays {
    #[must_use]
    pub fn new(maximum_allowed_size: u64) -> Self {
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
            if let ConstKind::Value(val) = cst.val;
            if let ConstValue::Scalar(element_count) = val;
            if let Ok(element_count) = element_count.to_machine_usize(&cx.tcx);
            if let Ok(element_size) = cx.layout_of(element_type).map(|l| l.size.bytes());
            if self.maximum_allowed_size < element_count * element_size;

            then {
                let hi_pos = item.ident.span.lo() - BytePos::from_usize(1);
                let sugg_span = Span::new(
                    hi_pos - BytePos::from_usize("const".len()),
                    hi_pos,
                    item.span.ctxt(),
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
                            "static".to_string(),
                            Applicability::MachineApplicable,
                        );
                    }
                );
            }
        }
    }
}
