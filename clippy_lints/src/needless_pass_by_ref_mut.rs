use super::needless_pass_by_value::requires_exact_signature;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{is_from_proc_macro, is_self};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, HirIdMap, HirIdSet, Impl, ItemKind, Mutability, Node, PatKind};
use rustc_hir_typeck::expr_use_visitor as euv;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::kw;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

declare_clippy_lint! {
    /// ### What it does
    /// Check if a `&mut` function argument is actually used mutably.
    ///
    /// Be careful if the function is publicly reexported as it would break compatibility with
    /// users of this function.
    ///
    /// ### Why is this bad?
    /// Less `mut` means less fights with the borrow checker. It can also lead to more
    /// opportunities for parallelization.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(y: &mut i32) -> i32 {
    ///     12 + *y
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo(y: &i32) -> i32 {
    ///     12 + *y
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_PASS_BY_REF_MUT,
    suspicious,
    "using a `&mut` argument when it's not mutated"
}

#[derive(Copy, Clone)]
pub struct NeedlessPassByRefMut {
    avoid_breaking_exported_api: bool,
}

impl NeedlessPassByRefMut {
    pub fn new(avoid_breaking_exported_api: bool) -> Self {
        Self {
            avoid_breaking_exported_api,
        }
    }
}

impl_lint_pass!(NeedlessPassByRefMut => [NEEDLESS_PASS_BY_REF_MUT]);

fn should_skip<'tcx>(
    cx: &LateContext<'tcx>,
    input: rustc_hir::Ty<'tcx>,
    ty: Ty<'_>,
    arg: &rustc_hir::Param<'_>,
) -> bool {
    // We check if this a `&mut`. `ref_mutability` returns `None` if it's not a reference.
    if !matches!(ty.ref_mutability(), Some(Mutability::Mut)) {
        return true;
    }

    if is_self(arg) {
        return true;
    }

    if let PatKind::Binding(.., name, _) = arg.pat.kind {
        // If it's a potentially unused variable, we don't check it.
        if name.name == kw::Underscore || name.as_str().starts_with('_') {
            return true;
        }
    }

    // All spans generated from a proc-macro invocation are the same...
    is_from_proc_macro(cx, &input)
}

impl<'tcx> LateLintPass<'tcx> for NeedlessPassByRefMut {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        fn_def_id: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        let hir_id = cx.tcx.hir().local_def_id_to_hir_id(fn_def_id);

        match kind {
            FnKind::ItemFn(.., header) => {
                let attrs = cx.tcx.hir().attrs(hir_id);
                if header.abi != Abi::Rust || requires_exact_signature(attrs) {
                    return;
                }
            },
            FnKind::Method(..) => (),
            FnKind::Closure => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir().find_parent(hir_id) {
            if matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            ) {
                return;
            }
        }

        let fn_sig = cx.tcx.fn_sig(fn_def_id).instantiate_identity();
        let fn_sig = cx.tcx.liberate_late_bound_regions(fn_def_id.to_def_id(), fn_sig);

        // If there are no `&mut` argument, no need to go any further.
        if !decl
            .inputs
            .iter()
            .zip(fn_sig.inputs())
            .zip(body.params)
            .any(|((&input, &ty), arg)| !should_skip(cx, input, ty, arg))
        {
            return;
        }

        // Collect variables mutably used and spans which will need dereferencings from the
        // function body.
        let MutablyUsedVariablesCtxt { mutably_used_vars, .. } = {
            let mut ctx = MutablyUsedVariablesCtxt::default();
            let infcx = cx.tcx.infer_ctxt().build();
            euv::ExprUseVisitor::new(&mut ctx, &infcx, fn_def_id, cx.param_env, cx.typeck_results()).consume_body(body);
            ctx
        };

        let mut it = decl
            .inputs
            .iter()
            .zip(fn_sig.inputs())
            .zip(body.params)
            .filter(|((&input, &ty), arg)| !should_skip(cx, input, ty, arg))
            .peekable();
        if it.peek().is_none() {
            return;
        }
        let show_semver_warning = self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(fn_def_id);
        for ((&input, &_), arg) in it {
            // Only take `&mut` arguments.
            if_chain! {
                if let PatKind::Binding(_, canonical_id, ..) = arg.pat.kind;
                if !mutably_used_vars.contains(&canonical_id);
                if let rustc_hir::TyKind::Ref(_, inner_ty) = input.kind;
                then {
                    // If the argument is never used mutably, we emit the warning.
                    let sp = input.span;
                    span_lint_and_then(
                        cx,
                        NEEDLESS_PASS_BY_REF_MUT,
                        sp,
                        "this argument is a mutable reference, but not used mutably",
                        |diag| {
                            diag.span_suggestion(
                                sp,
                                "consider changing to".to_string(),
                                format!(
                                    "&{}",
                                    snippet(cx, cx.tcx.hir().span(inner_ty.ty.hir_id), "_"),
                                ),
                                Applicability::Unspecified,
                            );
                            if show_semver_warning {
                                diag.warn("changing this function will impact semver compatibility");
                            }
                        },
                    );
                }
            }
        }
    }
}

#[derive(Default)]
struct MutablyUsedVariablesCtxt {
    mutably_used_vars: HirIdSet,
    prev_bind: Option<HirId>,
    aliases: HirIdMap<HirId>,
}

impl MutablyUsedVariablesCtxt {
    fn add_mutably_used_var(&mut self, mut used_id: HirId) {
        while let Some(id) = self.aliases.get(&used_id) {
            self.mutably_used_vars.insert(used_id);
            used_id = *id;
        }
        self.mutably_used_vars.insert(used_id);
    }
}

impl<'tcx> euv::Delegate<'tcx> for MutablyUsedVariablesCtxt {
    fn consume(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, _id: HirId) {
        if let euv::Place {
            base: euv::PlaceBase::Local(vid),
            base_ty,
            ..
        } = &cmt.place
        {
            if let Some(bind_id) = self.prev_bind.take() {
                self.aliases.insert(bind_id, *vid);
            } else if matches!(base_ty.ref_mutability(), Some(Mutability::Mut)) {
                self.add_mutably_used_var(*vid);
            }
        }
    }

    fn borrow(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, _id: HirId, borrow: ty::BorrowKind) {
        self.prev_bind = None;
        if let euv::Place {
            base: euv::PlaceBase::Local(vid),
            base_ty,
            ..
        } = &cmt.place
        {
            // If this is a mutable borrow, it was obviously used mutably so we add it. However
            // for `UniqueImmBorrow`, it's interesting because if you do: `array[0] = value` inside
            // a closure, it'll return this variant whereas if you have just an index access, it'll
            // return `ImmBorrow`. So if there is "Unique" and it's a mutable reference, we add it
            // to the mutably used variables set.
            if borrow == ty::BorrowKind::MutBorrow
                || (borrow == ty::BorrowKind::UniqueImmBorrow && base_ty.ref_mutability() == Some(Mutability::Mut))
            {
                self.add_mutably_used_var(*vid);
            }
        }
    }

    fn mutate(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, _id: HirId) {
        self.prev_bind = None;
        if let euv::Place {
            projections,
            base: euv::PlaceBase::Local(vid),
            ..
        } = &cmt.place
        {
            if !projections.is_empty() {
                self.add_mutably_used_var(*vid);
            }
        }
    }

    fn copy(&mut self, _cmt: &euv::PlaceWithHirId<'tcx>, _id: HirId) {
        self.prev_bind = None;
    }

    fn fake_read(&mut self, _: &rustc_hir_typeck::expr_use_visitor::PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}

    fn bind(&mut self, _cmt: &euv::PlaceWithHirId<'tcx>, id: HirId) {
        self.prev_bind = Some(id);
    }
}
