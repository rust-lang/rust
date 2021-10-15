use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_lint_defs::Applicability;
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

//
// TODO: Register the lint pass in `clippy_lints/src/lib.rs`,
//       e.g. store.register_early_pass(||
// Box::new(trailing_zero_sized_array_without_repr_c::TrailingZeroSizedArrayWithoutReprC));

impl EarlyLintPass for TrailingZeroSizedArrayWithoutReprC {
    fn check_struct_def(&mut self, cx: &EarlyContext<'_>, data: &rustc_ast::VariantData) {
        if is_struct_with_trailing_zero_sized_array(cx, data) && !has_repr_c(cx, data) {
            span_lint_and_sugg(
                cx,
                todo!(),
                todo!(),
                todo!(),
                "try",
                "`#[repr(C)]`".to_string(),
                Applicability::MachineApplicable,
            )
        }
    }
}

fn is_struct_with_trailing_zero_sized_array(cx: &EarlyContext<'_>, data: &rustc_ast::VariantData) -> bool {
    if_chain! {
        if let rustc_ast::ast::VariantData::Struct(field_defs, some_bool_huh) = data;
        if let Some(last_field) = field_defs.last();
        if let rustc_ast::ast::TyKind::Array(_, aconst) = &last_field.ty.kind;
        // TODO: if array is zero-sized;
        then {
            dbg!(aconst);
            true
        } else {
            false
        }
    }
}

fn has_repr_c(cx: &EarlyContext<'_>, data: &rustc_ast::VariantData) -> bool {
    todo!()
}
