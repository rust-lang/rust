use clippy_utils::consts::{miri_to_const, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Item, ItemKind, TyKind, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Displays a warning when a struct with a trailing zero-sized array is declared without the `repr(C)` attribute.
    ///
    /// ### Why is this bad?
    /// Zero-sized arrays aren't very useful in Rust itself, so such a struct is likely being created to pass to C code (or in conjuction with manual allocation to make it easy to compute the offset of the array). Either way, `#[repr(C)]` is needed.
    ///
    /// ### Example
    /// ```rust
    /// struct RarelyUseful {
    ///     some_field: usize,
    ///     last: [SomeType; 0],
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// #[repr(C)]
    /// struct MoreOftenUseful {
    ///     some_field: usize,
    ///     last: [SomeType; 0],
    /// }
    /// ```
    pub TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C,
    nursery,
    "struct with a trailing zero-sized array but without `repr(C)`"
}
declare_lint_pass!(TrailingZeroSizedArrayWithoutReprC => [TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C]);

//
// TODO: Register the lint pass in `clippy_lints/src/lib.rs`,
//       e.g. store.register_early_pass(||
// Box::new(trailing_zero_sized_array_without_repr_c::TrailingZeroSizedArrayWithoutReprC));
// DONE!

impl<'tcx> LateLintPass<'tcx> for TrailingZeroSizedArrayWithoutReprC {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if is_struct_with_trailing_zero_sized_array(cx, item) && !has_repr_c(cx, item.def_id) {
            span_lint_and_sugg(
                cx,
                TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C,
                item.span,
                "trailing zero-sized array in a struct which is not marked `#[repr(C)]`",
                "try annotating the struct definition with `#[repr(C)]` (or another `repr` attribute):",
                format!("#[repr(C)]\n{}", snippet(cx, item.span, "..")),
                Applicability::MaybeIncorrect,
            );
        }
    }
}

fn is_struct_with_trailing_zero_sized_array(cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) -> bool {
    if_chain! {
        if let ItemKind::Struct(data, _generics) = &item.kind;
        if let VariantData::Struct(field_defs, _) = data;
        if let Some(last_field) = field_defs.last();
        if let TyKind::Array(_, aconst) = last_field.ty.kind;
        let aconst_def_id = cx.tcx.hir().body_owner_def_id(aconst.body).to_def_id();
        let ty = cx.tcx.type_of(aconst_def_id);
        let constant = cx
            .tcx
            // NOTE: maybe const_eval_resolve?
            .const_eval_poly(aconst_def_id)
            .ok()
            .map(|val| rustc_middle::ty::Const::from_value(cx.tcx, val, ty));
        if let Some(Constant::Int(val)) = constant.and_then(miri_to_const);
        if val == 0;
        then {
            true
        } else {
            false
        }
    }
}

fn has_repr_c(cx: &LateContext<'tcx>, def_id: LocalDefId) -> bool {
    let hir_map = cx.tcx.hir();
    let hir_id = hir_map.local_def_id_to_hir_id(def_id);
    let attrs = hir_map.attrs(hir_id);

    // NOTE: Can there ever be more than one `repr` attribute?
    // other `repr` syms: repr, repr128, repr_align, repr_align_enum, repr_no_niche, repr_packed,
    // repr_simd, repr_transparent
    if let Some(_attr) = attrs.iter().find(|attr| attr.has_name(sym::repr)) {
        true
    } else {
        false
    }
}
