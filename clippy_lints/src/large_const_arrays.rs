use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::impl_lint_pass;
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
    maximum_allowed_size: u64,
}

impl LargeConstArrays {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            maximum_allowed_size: conf.array_size_threshold,
        }
    }
}

impl_lint_pass!(LargeConstArrays => [LARGE_CONST_ARRAYS]);

impl<'tcx> LateLintPass<'tcx> for LargeConstArrays {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Const(ident, generics, _, _) = &item.kind
            // Since static items may not have generics, skip generic const items.
            // FIXME(generic_const_items): I don't think checking `generics.hwcp` suffices as it
            // doesn't account for empty where-clauses that only consist of keyword `where` IINM.
            && generics.params.is_empty() && !generics.has_where_clause_predicates
            && !item.span.from_expansion()
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && let ty::Array(element_type, cst) = ty.kind()
            && let Some(element_count) = cx.tcx
                .try_normalize_erasing_regions(cx.typing_env(), *cst).unwrap_or(*cst).try_to_target_usize(cx.tcx)
            && let Ok(element_size) = cx.layout_of(*element_type).map(|l| l.size.bytes())
            && u128::from(self.maximum_allowed_size) < u128::from(element_count) * u128::from(element_size)
        {
            let hi_pos = ident.span.lo() - BytePos::from_usize(1);
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
                },
            );
        }
    }
}
