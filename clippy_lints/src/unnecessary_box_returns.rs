use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{def_id::LocalDefId, intravisit::FnKind, Body, FnDecl, FnRetTy};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for a return type containing a `Box<T>` where `T` implements `Sized`
    ///
    /// ### Why is this bad?
    ///
    /// It's better to just return `T` in these cases. The caller may not need
    /// the value to be boxed, and it's expensive to free the memory once the
    /// `Box<T>` been dropped.
    ///
    /// ### Example
    /// ```rust
    /// fn foo() -> Box<String> {
    ///     Box::new(String::from("Hello, world!"))
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo() -> String {
    ///     String::from("Hello, world!")
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub UNNECESSARY_BOX_RETURNS,
    pedantic,
    "Needlessly returning a Box"
}
declare_lint_pass!(UnnecessaryBoxReturns => [UNNECESSARY_BOX_RETURNS]);

impl LateLintPass<'_> for UnnecessaryBoxReturns {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        fn_kind: FnKind<'_>,
        decl: &FnDecl<'_>,
        _: &Body<'_>,
        _: Span,
        def_id: LocalDefId,
    ) {
        // it's unclear what part of a closure you would span, so for now it's ignored
        // if this is changed, please also make sure not to call `hir_ty_to_ty` below
        if matches!(fn_kind, FnKind::Closure) {
            return;
        }

        let FnRetTy::Return(return_ty_hir) = &decl.output else { return };

        let return_ty = cx
            .tcx
            .erase_late_bound_regions(cx.tcx.fn_sig(def_id).skip_binder())
            .output();

        if !return_ty.is_box() {
            return;
        }

        let boxed_ty = return_ty.boxed_ty();

        // it's sometimes useful to return Box<T> if T is unsized, so don't lint those
        if boxed_ty.is_sized(cx.tcx, cx.param_env) {
            span_lint_and_then(
                cx,
                UNNECESSARY_BOX_RETURNS,
                return_ty_hir.span,
                format!("boxed return of the sized type `{boxed_ty}`").as_str(),
                |diagnostic| {
                    diagnostic.span_suggestion(
                        return_ty_hir.span,
                        "try",
                        boxed_ty.to_string(),
                        // the return value and function callers also needs to
                        // be changed, so this can't be MachineApplicable
                        Applicability::Unspecified,
                    );
                    diagnostic.help("changing this also requires a change to the return expressions in this function");
                },
            );
        }
    }
}
