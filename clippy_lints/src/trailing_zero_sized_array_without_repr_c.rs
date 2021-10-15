use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_hir::*;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_lint::{LateContext, LateLintPass};
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
    /// struct MakesSense {
    ///     some_field: usize,
    ///     last: [SomeType; 0],
    /// }
    /// ```
    pub TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C,
    nursery,
    "struct with a trailing zero-sized array but without `repr(C)`"
}
declare_lint_pass!(TrailingZeroSizedArrayWithoutReprC => [TRAILING_ZERO_SIZED_ARRAY_WITHOUT_REPR_C]);

impl LateLintPass<'_> for TrailingZeroSizedArrayWithoutReprC {
    fn check_struct_def(&mut self, cx: &LateContext<'tcx>, data: &'tcx rustc_hir::VariantData<'tcx>) {
        dbg!("in check_struct_def");
        if_chain! {
            if let Some(def) = data.fields().last();
            if let rustc_hir::TyKind::Array(ty, acost) = def.ty.kind;
            then {
                // is the AnonConst `0`
            }
        }

        // span_lint_and_sugg(
        //     cx,
        //     todo!(),
        //     todo!(),
        //     todo!(),
        //     todo!(),
        //     todo!(),
        //     rustc_errors::Applicability::MaybeIncorrect,
        // )
    }
    // https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/sty/enum.TyKind.html#variant.Array in latepass
    // or https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.TyKind.html#variant.Array in early pass

    // fn check_struct_def_post(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_hir::VariantData<'tcx>)
    // {}

    // fn check_field_def(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_hir::FieldDef<'tcx>) {}

    // fn check_attribute(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_ast::Attribute) {}

    // fn enter_lint_attrs(&mut self, _: &LateContext<'tcx>, _: &'tcx [rustc_ast::Attribute]) {}

    // fn exit_lint_attrs(&mut self, _: &LateContext<'tcx>, _: &'tcx [rustc_ast::Attribute]) {}
}
//
// TODO: Register the lint pass in `clippy_lints/src/lib.rs`,
//       e.g. store.register_late_pass(||
// Box::new(trailing_zero_sized_array_without_repr_c::TrailingZeroSizedArrayWithoutReprC));

// fn temp_alert() {}

impl EarlyLintPass for TrailingZeroSizedArrayWithoutReprC {
    fn check_struct_def(&mut self, cx: &EarlyContext<'_>, data: &rustc_ast::VariantData) {
        if_chain! {
            if let rustc_ast::ast::VariantData::Struct(field_defs, some_bool_huh) = data;
            if let Some(last_field) = field_defs.last();
            if let rustc_ast::ast::TyKind::Array(_, aconst) = &last_field.ty.kind;            
            then {dbg!(aconst); return ();}
        }
    }
}
