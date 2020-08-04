use crate::utils::span_lint_and_help;
use if_chain::if_chain;
use rustc_ast::ast::{Param, TyKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** The lint checks for `self` fn fn parameters that explicitly
    /// specify the `Self`-type explicitly
    /// **Why is this bad?** Increases the amount and decreases the readability of code
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// enum ValType {
    ///     I32,
    ///     I64,
    ///     F32,
    ///     F64,
    /// }
    ///
    /// impl ValType {
    ///     pub fn bytes(self: Self) -> usize {
    ///         match self {
    ///             Self::I32 | Self::F32 => 4,
    ///             Self::I64 | Self::F64 => 8,
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Could be rewritten as
    ///
    /// ```rust
    /// enum ValType {
    ///     I32,
    ///     I64,
    ///     F32,
    ///     F64,
    /// }
    ///
    /// impl ValType {
    ///     pub fn bytes(self) -> usize {
    ///         match self {
    ///             Self::I32 | Self::F32 => 4,
    ///             Self::I64 | Self::F64 => 8,
    ///         }
    ///     }
    /// }
    /// ```
    pub NEEDLESS_FN_SELF_TYPE,
    style,
    "type of `self` parameter is already by default `Self`"
}

declare_lint_pass!(NeedlessFnSelfType => [NEEDLESS_FN_SELF_TYPE]);

impl EarlyLintPass for NeedlessFnSelfType {
    fn check_param(&mut self, cx: &EarlyContext<'_>, p: &Param) {
        if_chain! {
            if p.is_self();
            if let TyKind::Path(None, path) = &p.ty.kind;
            if let Some(segment) = path.segments.first();
            if segment.ident.as_str() == sym!(Self).as_str();
            then {
                span_lint_and_help(
                    cx,
                    NEEDLESS_FN_SELF_TYPE,
                    p.ty.span,
                    "the type of the `self` parameter is already by default `Self`",
                    None,
                    "consider removing the type specification",
                );
            }
        }
    }
}
