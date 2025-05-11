use clippy_utils::diagnostics::span_lint_and_then;
use rustc_attr_data_structures::{AttributeKind, ReprAttr, find_attr};
use rustc_hir::{HirId, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, FieldDef};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Displays a warning when a union is declared with the default representation (without a `#[repr(C)]` attribute).
    ///
    /// ### Why restrict this?
    /// Unions in Rust have unspecified layout by default, despite many people thinking that they
    /// lay out each field at the start of the union (like C does). That is, there are no guarantees
    /// about the offset of the fields for unions with multiple non-ZST fields without an explicitly
    /// specified layout. These cases may lead to undefined behavior in unsafe blocks.
    ///
    /// ### Example
    /// ```no_run
    /// union Foo {
    ///     a: i32,
    ///     b: u32,
    /// }
    ///
    /// fn main() {
    ///     let _x: u32 = unsafe {
    ///         Foo { a: 0_i32 }.b // Undefined behavior: `b` is allowed to be padding
    ///     };
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[repr(C)]
    /// union Foo {
    ///     a: i32,
    ///     b: u32,
    /// }
    ///
    /// fn main() {
    ///     let _x: u32 = unsafe {
    ///         Foo { a: 0_i32 }.b // Now defined behavior, this is just an i32 -> u32 transmute
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
        if !item.span.from_expansion()
            && is_union_with_two_non_zst_fields(cx, item)
            && !has_c_repr_attr(cx, item.hir_id())
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                DEFAULT_UNION_REPRESENTATION,
                item.span,
                "this union has the default representation",
                |diag| {
                    diag.help(format!(
                        "consider annotating `{}` with `#[repr(C)]` to explicitly specify memory layout",
                        cx.tcx.def_path_str(item.owner_id)
                    ));
                },
            );
        }
    }
}

/// Returns true if the given item is a union with at least two non-ZST fields.
/// (ZST fields having an arbitrary offset is completely inconsequential, and
/// if there is only one field left after ignoring ZST fields then the offset
/// of that field does not matter either.)
fn is_union_with_two_non_zst_fields<'tcx>(cx: &LateContext<'tcx>, item: &Item<'tcx>) -> bool {
    if let ItemKind::Union(..) = &item.kind
        && let ty::Adt(adt_def, args) = cx.tcx.type_of(item.owner_id).instantiate_identity().kind()
    {
        adt_def.all_fields().filter(|f| !is_zst(cx, f, args)).count() >= 2
    } else {
        false
    }
}

fn is_zst<'tcx>(cx: &LateContext<'tcx>, field: &FieldDef, args: ty::GenericArgsRef<'tcx>) -> bool {
    let ty = field.ty(cx.tcx, args);
    if let Ok(layout) = cx.layout_of(ty) {
        layout.is_zst()
    } else {
        false
    }
}

fn has_c_repr_attr(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    let attrs = cx.tcx.hir_attrs(hir_id);

    find_attr!(attrs, AttributeKind::Repr(r) if r.iter().any(|(x, _)| *x == ReprAttr::ReprC))
}
