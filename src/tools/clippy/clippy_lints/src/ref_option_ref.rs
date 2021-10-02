use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::last_path_segment;
use clippy_utils::source::snippet;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{GenericArg, Mutability, Ty, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `&Option<&T>`.
    ///
    /// ### Why is this bad?
    /// Since `&` is Copy, it's useless to have a
    /// reference on `Option<&T>`.
    ///
    /// ### Known problems
    /// It may be irrelevant to use this lint on
    /// public API code as it will make a breaking change to apply it.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let x: &Option<&u32> = &Some(&0u32);
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let x: Option<&u32> = Some(&0u32);
    /// ```
    pub REF_OPTION_REF,
    pedantic,
    "use `Option<&T>` instead of `&Option<&T>`"
}

declare_lint_pass!(RefOptionRef => [REF_OPTION_REF]);

impl<'tcx> LateLintPass<'tcx> for RefOptionRef {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx>) {
        if_chain! {
            if let TyKind::Rptr(_, ref mut_ty) = ty.kind;
            if mut_ty.mutbl == Mutability::Not;
            if let TyKind::Path(ref qpath) = &mut_ty.ty.kind;
            let last = last_path_segment(qpath);
            if let Some(res) = last.res;
            if let Some(def_id) = res.opt_def_id();

            if cx.tcx.is_diagnostic_item(sym::Option, def_id);
            if let Some(params) = last_path_segment(qpath).args ;
            if !params.parenthesized;
            if let Some(inner_ty) = params.args.iter().find_map(|arg| match arg {
                GenericArg::Type(inner_ty) => Some(inner_ty),
                _ => None,
            });
            if let TyKind::Rptr(_, _) = inner_ty.kind;

            then {
                span_lint_and_sugg(
                    cx,
                    REF_OPTION_REF,
                    ty.span,
                    "since `&` implements the `Copy` trait, `&Option<&T>` can be simplified to `Option<&T>`",
                    "try",
                    format!("Option<{}>", &snippet(cx, inner_ty.span, "..")),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
