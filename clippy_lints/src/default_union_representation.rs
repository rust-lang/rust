use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{self as hir, HirId, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use rustc_typeck::hir_ty_to_ty;

declare_clippy_lint! {
    /// ### What it does
    /// Displays a warning when a union is declared with the default representation (without a `#[repr(C)]` attribute).
    ///
    /// ### Why is this bad?
    /// Unions in Rust have unspecified layout by default, despite many people thinking that they
    /// lay out each field at the start of the union (like C does). That is, there are no guarantees
    /// about the offset of the fields for unions with multiple non-ZST fields without an explicitly
    /// specified layout. These cases may lead to undefined behavior in unsafe blocks.
    ///
    /// ### Example
    /// ```rust
    /// union Foo {
    ///     a: i32,
    ///     b: u32,
    /// }
    ///
    /// fn main() {
    ///     let _x: u32 = unsafe {
    ///         Foo { a: 0_i32 }.b // Undefined behaviour: `b` is allowed to be padding
    ///     };
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// #[repr(C)]
    /// union Foo {
    ///     a: i32,
    ///     b: u32,
    /// }
    ///
    /// fn main() {
    ///     let _x: u32 = unsafe {
    ///         Foo { a: 0_i32 }.b // Now defined behaviour, this is just an i32 -> u32 transmute
    ///     };
    /// }
    /// ```
    #[clippy::version = "1.60.0"]
    pub DEFAULT_UNION_REPRESENTATION,
    restriction,
    "unions without a `#[repr(C)]` attribute"
}
declare_lint_pass!(DefaultUnionRepresentation => [DEFAULT_UNION_REPRESENTATION]);

impl<'tcx> LateLintPass<'tcx> for DefaultUnionRepresentation {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if is_union_with_two_non_zst_fields(cx, item) && !has_c_repr_attr(cx, item.hir_id()) {
            span_lint_and_help(
                cx,
                DEFAULT_UNION_REPRESENTATION,
                item.span,
                "this union has the default representation",
                None,
                &format!(
                    "consider annotating `{}` with `#[repr(C)]` to explicitly specify memory layout",
                    cx.tcx.def_path_str(item.def_id.to_def_id())
                ),
            );
        }
    }
}

/// Returns true if the given item is a union with at least two non-ZST fields.
fn is_union_with_two_non_zst_fields(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Union(data, _) = &item.kind {
        data.fields().iter().filter(|f| !is_zst(cx, f.ty)).count() >= 2
    } else {
        false
    }
}

fn is_zst(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>) -> bool {
    if hir_ty.span.from_expansion() {
        return false;
    }
    let ty = hir_ty_to_ty(cx.tcx, hir_ty);
    if let Ok(layout) = cx.layout_of(ty) {
        layout.is_zst()
    } else {
        false
    }
}

fn has_c_repr_attr(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    cx.tcx.hir().attrs(hir_id).iter().any(|attr| {
        if attr.has_name(sym::repr) {
            if let Some(items) = attr.meta_item_list() {
                for item in items {
                    if item.is_word() && matches!(item.name_or_empty(), sym::C) {
                        return true;
                    }
                }
            }
        }
        false
    })
}
