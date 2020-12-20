use crate::utils::qualify_min_const_fn::is_min_const_fn;
use crate::utils::{
    fn_has_unsatisfiable_preds, has_drop, is_entrypoint_fn, meets_msrv, span_lint, trait_ref_of_method,
};
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, Constness, FnDecl, GenericParamKind, HirId};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use rustc_typeck::hir_ty_to_ty;

const MISSING_CONST_FOR_FN_MSRV: RustcVersion = RustcVersion::new(1, 37, 0);

declare_clippy_lint! {
    /// **What it does:**
    ///
    /// Suggests the use of `const` in functions and methods where possible.
    ///
    /// **Why is this bad?**
    ///
    /// Not having the function const prevents callers of the function from being const as well.
    ///
    /// **Known problems:**
    ///
    /// Const functions are currently still being worked on, with some features only being available
    /// on nightly. This lint does not consider all edge cases currently and the suggestions may be
    /// incorrect if you are using this lint on stable.
    ///
    /// Also, the lint only runs one pass over the code. Consider these two non-const functions:
    ///
    /// ```rust
    /// fn a() -> i32 {
    ///     0
    /// }
    /// fn b() -> i32 {
    ///     a()
    /// }
    /// ```
    ///
    /// When running Clippy, the lint will only suggest to make `a` const, because `b` at this time
    /// can't be const as it calls a non-const function. Making `a` const and running Clippy again,
    /// will suggest to make `b` const, too.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # struct Foo {
    /// #     random_number: usize,
    /// # }
    /// # impl Foo {
    /// fn new() -> Self {
    ///     Self { random_number: 42 }
    /// }
    /// # }
    /// ```
    ///
    /// Could be a const fn:
    ///
    /// ```rust
    /// # struct Foo {
    /// #     random_number: usize,
    /// # }
    /// # impl Foo {
    /// const fn new() -> Self {
    ///     Self { random_number: 42 }
    /// }
    /// # }
    /// ```
    pub MISSING_CONST_FOR_FN,
    nursery,
    "Lint functions definitions that could be made `const fn`"
}

impl_lint_pass!(MissingConstForFn => [MISSING_CONST_FOR_FN]);

pub struct MissingConstForFn {
    msrv: Option<RustcVersion>,
}

impl MissingConstForFn {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for MissingConstForFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        kind: FnKind<'_>,
        _: &FnDecl<'_>,
        _: &Body<'_>,
        span: Span,
        hir_id: HirId,
    ) {
        if !meets_msrv(self.msrv.as_ref(), &MISSING_CONST_FOR_FN_MSRV) {
            return;
        }

        let def_id = cx.tcx.hir().local_def_id(hir_id);

        if in_external_macro(cx.tcx.sess, span) || is_entrypoint_fn(cx, def_id.to_def_id()) {
            return;
        }

        // Building MIR for `fn`s with unsatisfiable preds results in ICE.
        if fn_has_unsatisfiable_preds(cx, def_id.to_def_id()) {
            return;
        }

        // Perform some preliminary checks that rule out constness on the Clippy side. This way we
        // can skip the actual const check and return early.
        match kind {
            FnKind::ItemFn(_, generics, header, ..) => {
                let has_const_generic_params = generics
                    .params
                    .iter()
                    .any(|param| matches!(param.kind, GenericParamKind::Const { .. }));

                if already_const(header) || has_const_generic_params {
                    return;
                }
            },
            FnKind::Method(_, sig, ..) => {
                if trait_ref_of_method(cx, hir_id).is_some()
                    || already_const(sig.header)
                    || method_accepts_dropable(cx, sig.decl.inputs)
                {
                    return;
                }
            },
            FnKind::Closure(..) => return,
        }

        let mir = cx.tcx.optimized_mir(def_id);

        if let Err((span, err)) = is_min_const_fn(cx.tcx, &mir) {
            if rustc_mir::const_eval::is_min_const_fn(cx.tcx, def_id.to_def_id()) {
                cx.tcx.sess.span_err(span, &err);
            }
        } else {
            span_lint(cx, MISSING_CONST_FOR_FN, span, "this could be a `const fn`");
        }
    }
    extract_msrv_attr!(LateContext);
}

/// Returns true if any of the method parameters is a type that implements `Drop`. The method
/// can't be made const then, because `drop` can't be const-evaluated.
fn method_accepts_dropable(cx: &LateContext<'_>, param_tys: &[hir::Ty<'_>]) -> bool {
    // If any of the params are droppable, return true
    param_tys.iter().any(|hir_ty| {
        let ty_ty = hir_ty_to_ty(cx.tcx, hir_ty);
        has_drop(cx, ty_ty)
    })
}

// We don't have to lint on something that's already `const`
#[must_use]
fn already_const(header: hir::FnHeader) -> bool {
    header.constness == Constness::Const
}
