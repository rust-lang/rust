use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::qualify_min_const_fn::is_min_const_fn;
use clippy_utils::{fn_has_unsatisfiable_preds, is_entrypoint_fn, is_from_proc_macro, is_in_test, trait_ref_of_method};
use rustc_abi::ExternAbi;
use rustc_errors::Applicability;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{self as hir, Body, Constness, FnDecl, GenericParamKind, OwnerId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// Suggests the use of `const` in functions and methods where possible.
    ///
    /// ### Why is this bad?
    /// Not having the function const prevents callers of the function from being const as well.
    ///
    /// ### Known problems
    /// Const functions are currently still being worked on, with some features only being available
    /// on nightly. This lint does not consider all edge cases currently and the suggestions may be
    /// incorrect if you are using this lint on stable.
    ///
    /// Also, the lint only runs one pass over the code. Consider these two non-const functions:
    ///
    /// ```no_run
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
    /// If you are marking a public function with `const`, removing it again will break API compatibility.
    /// ### Example
    /// ```no_run
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
    /// ```no_run
    /// # struct Foo {
    /// #     random_number: usize,
    /// # }
    /// # impl Foo {
    /// const fn new() -> Self {
    ///     Self { random_number: 42 }
    /// }
    /// # }
    /// ```
    #[clippy::version = "1.34.0"]
    pub MISSING_CONST_FOR_FN,
    nursery,
    "Lint functions definitions that could be made `const fn`"
}

impl_lint_pass!(MissingConstForFn => [MISSING_CONST_FOR_FN]);

pub struct MissingConstForFn {
    msrv: Msrv,
}

impl MissingConstForFn {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for MissingConstForFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _: &FnDecl<'_>,
        body: &Body<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) {
        let hir_id = cx.tcx.local_def_id_to_hir_id(def_id);
        if is_in_test(cx.tcx, hir_id) {
            return;
        }

        if span.in_external_macro(cx.tcx.sess.source_map()) || is_entrypoint_fn(cx, def_id.to_def_id()) {
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

                if already_const(header)
                    || has_const_generic_params
                    || !could_be_const_with_abi(cx, self.msrv, header.abi)
                {
                    return;
                }
            },
            FnKind::Method(_, sig, ..) => {
                if already_const(sig.header) || trait_ref_of_method(cx, OwnerId { def_id }).is_some() {
                    return;
                }
            },
            FnKind::Closure => return,
        }

        if fn_inputs_has_impl_trait_ty(cx, def_id) {
            return;
        }

        // Const fns are not allowed as methods in a trait.
        {
            let parent = cx.tcx.hir_get_parent_item(hir_id).def_id;
            if parent != CRATE_DEF_ID
                && let hir::Node::Item(item) = cx.tcx.hir_node_by_def_id(parent)
                && let hir::ItemKind::Trait(..) = &item.kind
            {
                return;
            }
        }

        if !self.msrv.meets(cx, msrvs::CONST_IF_MATCH) {
            return;
        }

        if is_from_proc_macro(cx, &(&kind, body, hir_id, span)) {
            return;
        }

        let mir = cx.tcx.optimized_mir(def_id);

        if let Ok(()) = is_min_const_fn(cx, mir, self.msrv)
            && let hir::Node::Item(hir::Item { vis_span, .. }) | hir::Node::ImplItem(hir::ImplItem { vis_span, .. }) =
                cx.tcx.hir_node_by_def_id(def_id)
        {
            let suggestion = if vis_span.is_empty() { "const " } else { " const" };
            span_lint_and_then(cx, MISSING_CONST_FOR_FN, span, "this could be a `const fn`", |diag| {
                diag.span_suggestion_verbose(
                    vis_span.shrink_to_hi(),
                    "make the function `const`",
                    suggestion,
                    Applicability::MachineApplicable,
                );
            });
        }
    }
}

// We don't have to lint on something that's already `const`
#[must_use]
fn already_const(header: hir::FnHeader) -> bool {
    header.constness == Constness::Const
}

fn could_be_const_with_abi(cx: &LateContext<'_>, msrv: Msrv, abi: ExternAbi) -> bool {
    match abi {
        ExternAbi::Rust => true,
        // `const extern "C"` was stabilized after 1.62.0
        ExternAbi::C { unwind: false } => msrv.meets(cx, msrvs::CONST_EXTERN_C_FN),
        // Rest ABIs are still unstable and need the `const_extern_fn` feature enabled.
        _ => msrv.meets(cx, msrvs::CONST_EXTERN_FN),
    }
}

/// Return `true` when the given `def_id` is a function that has `impl Trait` ty as one of
/// its parameter types.
fn fn_inputs_has_impl_trait_ty(cx: &LateContext<'_>, def_id: LocalDefId) -> bool {
    let inputs = cx.tcx.fn_sig(def_id).instantiate_identity().inputs().skip_binder();
    inputs.iter().any(|input| {
        matches!(
            input.kind(),
            ty::Alias(ty::AliasTyKind::Free, alias_ty) if cx.tcx.type_of(alias_ty.def_id).skip_binder().is_impl_trait()
        )
    })
}
