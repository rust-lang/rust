use clippy_utils::consts::{miri_to_const, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use rustc_ast::Attribute;
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Item, ItemKind, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::dep_graph::DepContext;
use rustc_middle::ty as ty_mod;
use rustc_middle::ty::ReprFlags;
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
    "struct with a trailing zero-sized array but without `repr(C)` or another `repr` attribute"
}
declare_lint_pass!(TrailingZeroSizedArrayWithoutReprC => [TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C]);

// TESTNAME=trailing_zero_sized_array_without_repr_c cargo uitest

impl<'tcx> LateLintPass<'tcx> for TrailingZeroSizedArrayWithoutReprC {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        dbg!(item.ident);
        if is_struct_with_trailing_zero_sized_array(cx, item) && !has_repr_attr(cx, item.def_id) {
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
        if let rustc_hir::TyKind::Array(_, aconst) = last_field.ty.kind;
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

fn has_repr_attr(cx: &LateContext<'tcx>, def_id: LocalDefId) -> bool {
    let attrs_get_attrs = get_attrs_get_attrs(cx, def_id);
    let attrs_hir_map = get_attrs_hir_map(cx, def_id);
    let b11 = dbg!(includes_repr_attr_using_sym(attrs_get_attrs));
    let b12 = dbg!(includes_repr_attr_using_sym(attrs_hir_map));
    let b21 = dbg!(includes_repr_attr_using_helper(cx, attrs_get_attrs));
    let b22 = dbg!(includes_repr_attr_using_helper(cx, attrs_hir_map));
    let b3 = dbg!(has_repr_attr_using_adt(cx, def_id));
    let all_same = b11 && b12 && b21 && b22 && b3;
    dbg!(all_same);

    b11
}

fn get_attrs_get_attrs(cx: &LateContext<'tcx>, def_id: LocalDefId) -> &'tcx [Attribute] {
    cx.tcx.get_attrs(def_id.to_def_id())
}

fn get_attrs_hir_map(cx: &LateContext<'tcx>, def_id: LocalDefId) -> &'tcx [Attribute] {
    let hir_map = cx.tcx.hir();
    let hir_id = hir_map.local_def_id_to_hir_id(def_id);
    hir_map.attrs(hir_id)
}

// Don't like this because it's so dependent on the current list of `repr` flags and it would have to be manually updated if that ever expanded. idk if there's any mechanism in `bitflag!` or elsewhere for requiring that sort of exhaustiveness
fn has_repr_attr_using_adt(cx: &LateContext<'tcx>, def_id: LocalDefId) -> bool {
    let ty = cx.tcx.type_of(def_id.to_def_id());
    if let ty_mod::Adt(adt, _) = ty.kind() {
        if adt.is_struct() {
            let repr = adt.repr;
            let repr_attr = ReprFlags::IS_C | ReprFlags::IS_TRANSPARENT | ReprFlags::IS_SIMD | ReprFlags::IS_LINEAR;
            repr.int.is_some() || repr.align.is_some() || repr.pack.is_some() || repr.flags.intersects(repr_attr)
        } else {
            false
        }
    } else {
        false
    }
}

fn includes_repr_attr_using_sym(attrs: &'tcx [Attribute]) -> bool {
    attrs.iter().any(|attr| attr.has_name(sym::repr))
}

fn includes_repr_attr_using_helper(cx: &LateContext<'tcx>, attrs: &'tcx [Attribute]) -> bool {
    attrs.iter().any(|attr| !rustc_attr::find_repr_attrs(cx.tcx.sess(), attr).is_empty())
}
