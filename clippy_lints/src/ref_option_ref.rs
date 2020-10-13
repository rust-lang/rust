use crate::utils::{last_path_segment, match_def_path, paths, snippet, span_lint_and_sugg};
use rustc_hir::{GenericArg, Local, Mutability, Ty, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use if_chain::if_chain;
use rustc_errors::Applicability;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `&Option<&T>`.
    ///
    /// **Why is this bad?** Since `&` is Copy, it's useless to have a
    /// reference on `Option<&T>`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// // example code where clippy issues a warning
    /// let x: &Option<&u32> = &Some(&0u32);
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // example code which does not raise clippy warning
    /// let x: Option<&u32> = Some(&0u32);
    /// ```
    pub REF_OPTION_REF,
    style,
    "use `Option<&T>` instead of `&Option<&T>`"
}

declare_lint_pass!(RefOptionRef => [REF_OPTION_REF]);

impl<'tcx> LateLintPass<'tcx> for RefOptionRef {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'_>) {

        if let Some(ref ty) = local.ty {
            self.check_ref_option_ref(cx, ty);
        }
    }
}

impl RefOptionRef {
    fn check_ref_option_ref(&self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx>) {
        if_chain! {
            if let TyKind::Rptr(_, ref mut_ty) = ty.kind;
            if mut_ty.mutbl == Mutability::Not;
            if let TyKind::Path(ref qpath) = &mut_ty.ty.kind ;
            if let Some(def_id) = cx.typeck_results().qpath_res(qpath, ty.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::OPTION);
            if let Some(ref params) = last_path_segment(qpath).args ;
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
                    "since & implements Copy trait, &Option<&T> can be simplifyied into Option<&T>",
                    "try",
                    format!("Option<{}>", &snippet(cx, inner_ty.span, "..")),
                    Applicability::Unspecified,
                );
            }
        }
    }
}