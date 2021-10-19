use clippy_utils::{consts::miri_to_const, consts::Constant, diagnostics::span_lint_and_help};
use rustc_hir::{HirId, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Displays a warning when a struct with a trailing zero-sized array is declared without a `repr` attribute.
    ///
    /// ### Why is this bad?
    /// Zero-sized arrays aren't very useful in Rust itself, so such a struct is likely being created to pass to C code or in some other situation where control over memory layout matters (for example, in conjuction with manual allocation to make it easy to compute the offset of the array). Either way, `#[repr(C)]` (or another `repr` attribute) is needed.
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
    pub TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR,
    nursery,
    "struct with a trailing zero-sized array but without `#[repr(C)]` or another `repr` attribute"
}
declare_lint_pass!(TrailingZeroSizedArrayWithoutRepr => [TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR]);

impl<'tcx> LateLintPass<'tcx> for TrailingZeroSizedArrayWithoutRepr {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        dbg!(item.ident);
        if is_struct_with_trailing_zero_sized_array(cx, item) && !has_repr_attr(cx, item.hir_id()) {
            // span_lint_and_help(
            //     cx,
            //     TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR,
            //     item.span,
            //     "trailing zero-sized array in a struct which is not marked with a `repr` attribute",
            //     None,
            //     "",
            // );
            eprintln!("consider yourself linted 😎");
        }
    }
}

fn is_struct_with_trailing_zero_sized_array(cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) -> bool {
    // TODO: when finalized, replace with an `if_chain`. I have it like this because my rust-analyzer
    // doesn't work when it's an `if_chain`.

    // First check if last field is an array
    if let ItemKind::Struct(data, _) = &item.kind {
        if let Some(last_field) = data.fields().last() {
            if let rustc_hir::TyKind::Array(_, length) = last_field.ty.kind {
                let length_did = cx.tcx.hir().body_owner_def_id(length.body).to_def_id();
                let ty = cx.tcx.type_of(length_did);
                let length = cx
                    .tcx
                    // ICE happens in `const_eval_poly` according to my backtrace
                    .const_eval_poly(length_did)
                    .ok()
                    .map(|val| rustc_middle::ty::Const::from_value(cx.tcx, val, ty));
                if let Some(Constant::Int(length)) = length.and_then(miri_to_const){
                    length == 0
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

fn has_repr_attr(cx: &LateContext<'tcx>, hir_id: HirId) -> bool {
    // NOTE: there's at least four other ways to do this but I liked this one the best. (All five agreed
    // on all testcases.) Happy to use another;
    // they're in the commit history if you want to look (or I can go find them).
    cx.tcx.hir().attrs(hir_id).iter().any(|attr| attr.has_name(sym::repr))
}
