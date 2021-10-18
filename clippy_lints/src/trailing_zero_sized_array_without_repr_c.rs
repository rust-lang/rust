use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::Attribute;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::dep_graph::DepContext;
use rustc_middle::ty::Const;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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

impl<'tcx> LateLintPass<'tcx> for TrailingZeroSizedArrayWithoutReprC {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if is_struct_with_trailing_zero_sized_array(cx, item) {
            // NOTE: This is to include attributes on the definition when we print the lint. If the convention
            // is to not do that with struct definitions (I'm not sure), then this isn't necessary. (note: if
            // you don't get rid of this, change `has_repr_attr` to `includes_repr_attr`).
            let attrs = cx.tcx.get_attrs(item.def_id.to_def_id());
            let first_attr = attrs.iter().min_by_key(|attr| attr.span.lo());
            let lint_span = if let Some(first_attr) = first_attr {
                first_attr.span.to(item.span)
            } else {
                item.span
            };

            if !has_repr_attr(cx, attrs) {
                span_lint_and_help(
                    cx,
                    TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C,
                    lint_span,
                    "trailing zero-sized array in a struct which is not marked `#[repr(C)]`",
                    None,
                    "consider annotating the struct definition with `#[repr(C)]` (or another `repr` attribute)",
                );
            }
        }
    }
}

fn is_struct_with_trailing_zero_sized_array(cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) -> bool {
    // TODO: when finalized, replace with an `if_chain`. I have it like this because my rust-analyzer
    // doesn't work when it's an `if_chain` First check if last field is an array
    if let ItemKind::Struct(data, _) = &item.kind {
        let field_defs = data.fields();
        if let Some(last_field) = field_defs.last() {
            if let rustc_hir::TyKind::Array(_, length) = last_field.ty.kind {
                // Then check if that that array zero-sized
                let length_ldid = cx.tcx.hir().local_def_id(length.hir_id);
                let length = Const::from_anon_const(cx.tcx, length_ldid);
                let length = length.try_eval_usize(cx.tcx, cx.param_env);
                // if let Some((Constant::Int(length), _)) = length {
                if let Some(length) = length { length == 0 } else { false }
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

fn has_repr_attr(cx: &LateContext<'tcx>, attrs: &[Attribute]) -> bool {
    // NOTE: there's at least four other ways to do this but I liked this one the best. (All five agreed
    // on all testcases (when i wrote this comment. I added a few since then).) Happy to use another;
    // they're in the commit history if you want to look (or I can go find them).
    attrs
        .iter()
        .any(|attr| !rustc_attr::find_repr_attrs(cx.tcx.sess(), attr).is_empty())
}
