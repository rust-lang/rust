use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{has_repr_attr, is_in_test};
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Displays a warning when a struct with a trailing zero-sized array is declared without a `repr` attribute.
    ///
    /// ### Why is this bad?
    /// Zero-sized arrays aren't very useful in Rust itself, so such a struct is likely being created to pass to C code or in some other situation where control over memory layout matters (for example, in conjunction with manual allocation to make it easy to compute the offset of the array). Either way, `#[repr(C)]` (or another `repr` attribute) is needed.
    ///
    /// ### Example
    /// ```no_run
    /// struct RarelyUseful {
    ///     some_field: u32,
    ///     last: [u32; 0],
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[repr(C)]
    /// struct MoreOftenUseful {
    ///     some_field: usize,
    ///     last: [u32; 0],
    /// }
    /// ```
    #[clippy::version = "1.58.0"]
    pub TRAILING_EMPTY_ARRAY,
    nursery,
    "struct with a trailing zero-sized array but without `#[repr(C)]` or another `repr` attribute"
}
declare_lint_pass!(TrailingEmptyArray => [TRAILING_EMPTY_ARRAY]);

impl<'tcx> LateLintPass<'tcx> for TrailingEmptyArray {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if is_struct_with_trailing_zero_sized_array(cx, item)
            && !has_repr_attr(cx, item.hir_id())
            && !is_in_test(cx.tcx, item.hir_id())
        {
            span_lint_and_help(
                cx,
                TRAILING_EMPTY_ARRAY,
                item.span,
                "trailing zero-sized array in a struct which is not marked with a `repr` attribute",
                None,
                format!(
                    "consider annotating `{}` with `#[repr(C)]` or another `repr` attribute",
                    cx.tcx.def_path_str(item.owner_id)
                ),
            );
        }
    }
}

fn is_struct_with_trailing_zero_sized_array<'tcx>(cx: &LateContext<'tcx>, item: &Item<'tcx>) -> bool {
    if let ItemKind::Struct(_, _, data) = &item.kind
        && let Some(last_field) = data.fields().last()
        && let field_ty = cx.tcx.normalize_erasing_regions(
            cx.typing_env(),
            cx.tcx.type_of(last_field.def_id).instantiate_identity(),
        )
        && let ty::Array(_, array_len) = *field_ty.kind()
        && let Some(0) = array_len.try_to_target_usize(cx.tcx)
    {
        true
    } else {
        false
    }
}
