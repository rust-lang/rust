use super::needless_pass_by_value::requires_exact_signature;
use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{inherits_cfg, is_from_proc_macro, is_self};
use core::ops::ControlFlow;
use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    BlockCheckMode, Body, Closure, Expr, ExprKind, FnDecl, HirId, HirIdMap, HirIdSet, Impl, ItemKind, Mutability, Node,
    PatKind,
};
use rustc_hir_typeck::expr_use_visitor as euv;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, Ty, TyCtxt, UpvarId, UpvarPath};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// Check if a `&mut` function argument is actually used mutably.
    ///
    /// Be careful if the function is publicly reexported as it would break compatibility with
    /// users of this function, when the users pass this function as an argument.
    ///
    /// ### Why is this bad?
    /// Less `mut` means less fights with the borrow checker. It can also lead to more
    /// opportunities for parallelization.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(y: &mut i32) -> i32 {
    ///     12 + *y
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn foo(y: &i32) -> i32 {
    ///     12 + *y
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub NEEDLESS_PASS_BY_REF_MUT,
    nursery,
    "using a `&mut` argument when it's not mutated"
}

pub struct NeedlessPassByRefMut<'tcx> {
    avoid_breaking_exported_api: bool,
    used_fn_def_ids: FxHashSet<LocalDefId>,
    fn_def_ids_to_maybe_unused_mut: FxIndexMap<LocalDefId, Vec<rustc_hir::Ty<'tcx>>>,
}

impl NeedlessPassByRefMut<'_> {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            used_fn_def_ids: FxHashSet::default(),
            fn_def_ids_to_maybe_unused_mut: FxIndexMap::default(),
        }
    }
}

impl_lint_pass!(NeedlessPassByRefMut<'_> => [NEEDLESS_PASS_BY_REF_MUT]);

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
        // Interestingly enough, `self` arguments make `is_from_proc_macro` return `true`, hence why
        // we return early here.
        return false;
    }

    if let PatKind::Binding(.., name, _) = arg.pat.kind
        // If it's a potentially unused variable, we don't check it.
        && (name.name == kw::Underscore || name.as_str().starts_with('_'))
    {
        return true;
    }

    // All spans generated from a proc-macro invocation are the same...
    is_from_proc_macro(cx, &input)
}

fn check_closures<'tcx>(
    ctx: &mut MutablyUsedVariablesCtxt<'tcx>,
    cx: &LateContext<'tcx>,
    checked_closures: &mut FxHashSet<LocalDefId>,
    closures: FxIndexSet<LocalDefId>,
) {
    for closure in closures {
        if !checked_closures.insert(closure) {
            continue;
        }
        ctx.prev_bind = None;
        ctx.prev_move_to_closure.clear();
        if let Some(body) = cx
            .tcx
            .hir_node_by_def_id(closure)
            .associated_body()
            .map(|(_, body_id)| cx.tcx.hir_body(body_id))
        {
            euv::ExprUseVisitor::for_clippy(cx, closure, &mut *ctx)
                .consume_body(body)
                .into_ok();
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NeedlessPassByRefMut<'tcx> {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'_>,
        span: Span,
        fn_def_id: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(fn_def_id) {
            return;
        }

        let hir_id = cx.tcx.local_def_id_to_hir_id(fn_def_id);
        let is_async = match kind {
            FnKind::ItemFn(.., header) => {
                if header.is_unsafe() {
                    // We don't check unsafe functions.
                    return;
                }
                let attrs = cx.tcx.hir_attrs(hir_id);
                if header.abi != ExternAbi::Rust || requires_exact_signature(attrs) {
                    return;
                }
                header.is_async()
            },
            FnKind::Method(.., sig) => {
                if sig.header.is_unsafe() {
                    // We don't check unsafe functions.
                    return;
                }
                sig.header.is_async()
            },
            FnKind::Closure => return,
        };

        // Exclude non-inherent impls
        if let Node::Item(item) = cx.tcx.parent_hir_node(hir_id)
            && matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            )
        {
            return;
        }

        let fn_sig = cx.tcx.fn_sig(fn_def_id).instantiate_identity();
        let fn_sig = cx.tcx.liberate_late_bound_regions(fn_def_id.to_def_id(), fn_sig);

        // If there are no `&mut` argument, no need to go any further.
        let mut it = decl
            .inputs
            .iter()
            .zip(fn_sig.inputs())
            .zip(body.params)
            .filter(|&((&input, &ty), arg)| !should_skip(cx, input, ty, arg))
            .peekable();
        if it.peek().is_none() {
            return;
        }
        // Collect variables mutably used and spans which will need dereferencings from the
        // function body.
        let mutably_used_vars = {
            let mut ctx = MutablyUsedVariablesCtxt {
                mutably_used_vars: HirIdSet::default(),
                prev_bind: None,
                prev_move_to_closure: HirIdSet::default(),
                aliases: HirIdMap::default(),
                async_closures: FxIndexSet::default(),
                tcx: cx.tcx,
            };
            euv::ExprUseVisitor::for_clippy(cx, fn_def_id, &mut ctx)
                .consume_body(body)
                .into_ok();

            let mut checked_closures = FxHashSet::default();

            // We retrieve all the closures declared in the function because they will not be found
            // by `euv::Delegate`.
            let mut closures: FxIndexSet<LocalDefId> = FxIndexSet::default();
            for_each_expr(cx, body, |expr| {
                if let ExprKind::Closure(closure) = expr.kind {
                    closures.insert(closure.def_id);
                }
                ControlFlow::<()>::Continue(())
            });
            check_closures(&mut ctx, cx, &mut checked_closures, closures);

            if is_async {
                while !ctx.async_closures.is_empty() {
                    let async_closures = ctx.async_closures.clone();
                    ctx.async_closures.clear();
                    check_closures(&mut ctx, cx, &mut checked_closures, async_closures);
                }
            }
            ctx.generate_mutably_used_ids_from_aliases()
        };
        for ((&input, &_), arg) in it {
            // Only take `&mut` arguments.
            if let PatKind::Binding(_, canonical_id, ..) = arg.pat.kind
                && !mutably_used_vars.contains(&canonical_id)
            {
                self.fn_def_ids_to_maybe_unused_mut
                    .entry(fn_def_id)
                    .or_default()
                    .push(input);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        // #11182; do not lint if mutability is required elsewhere
        if let ExprKind::Path(..) = expr.kind
            && let ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(expr).kind()
            && let Some(def_id) = def_id.as_local()
        {
            if let Node::Expr(e) = cx.tcx.parent_hir_node(expr.hir_id)
                && let ExprKind::Call(call, _) = e.kind
                && call.hir_id == expr.hir_id
            {
                return;
            }

            // We don't need to check each argument individually as you cannot coerce a function
            // taking `&mut` -> `&`, for some reason, so if we've gotten this far we know it's
            // passed as a `fn`-like argument (or is unified) and should ignore every "unused"
            // argument entirely
            self.used_fn_def_ids.insert(def_id);
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for (fn_def_id, unused) in self
            .fn_def_ids_to_maybe_unused_mut
            .iter()
            .filter(|(def_id, _)| !self.used_fn_def_ids.contains(def_id))
        {
            let mut is_cfged = None;
            for input in unused {
                // If the argument is never used mutably, we emit the warning.
                let sp = input.span;
                if let rustc_hir::TyKind::Ref(_, inner_ty) = input.kind {
                    let is_cfged = is_cfged.get_or_insert_with(|| inherits_cfg(cx.tcx, *fn_def_id));
                    span_lint_hir_and_then(
                        cx,
                        NEEDLESS_PASS_BY_REF_MUT,
                        cx.tcx.local_def_id_to_hir_id(*fn_def_id),
                        sp,
                        "this argument is a mutable reference, but not used mutably",
                        |diag| {
                            diag.span_suggestion(
                                sp,
                                "consider changing to".to_string(),
                                format!("&{}", snippet(cx, cx.tcx.hir_span(inner_ty.ty.hir_id), "_"),),
                                Applicability::Unspecified,
                            );
                            if cx.effective_visibilities.is_exported(*fn_def_id) {
                                diag.warn("changing this function will impact semver compatibility");
                            }
                            if *is_cfged {
                                diag.note("this is cfg-gated and may require further changes");
                            }
                        },
                    );
                }
            }
        }
    }
}

struct MutablyUsedVariablesCtxt<'tcx> {
    mutably_used_vars: HirIdSet,
    prev_bind: Option<HirId>,
    /// In async functions, the inner AST is composed of multiple layers until we reach the code
    /// defined by the user. Because of that, some variables are marked as mutably borrowed even
    /// though they're not. This field lists the `HirId` that should not be considered as mutable
    /// use of a variable.
    prev_move_to_closure: HirIdSet,
    aliases: HirIdMap<HirId>,
    async_closures: FxIndexSet<LocalDefId>,
    tcx: TyCtxt<'tcx>,
}

impl MutablyUsedVariablesCtxt<'_> {
    fn add_mutably_used_var(&mut self, used_id: HirId) {
        self.mutably_used_vars.insert(used_id);
    }

    // Because the alias may come after the mutable use of a variable, we need to fill the map at
    // the end.
    fn generate_mutably_used_ids_from_aliases(mut self) -> HirIdSet {
        let all_ids = self.mutably_used_vars.iter().copied().collect::<Vec<_>>();
        for mut used_id in all_ids {
            while let Some(id) = self.aliases.get(&used_id) {
                self.mutably_used_vars.insert(used_id);
                used_id = *id;
            }
            self.mutably_used_vars.insert(used_id);
        }
        self.mutably_used_vars
    }

    fn would_be_alias_cycle(&self, alias: HirId, mut target: HirId) -> bool {
        while let Some(id) = self.aliases.get(&target) {
            if *id == alias {
                return true;
            }
            target = *id;
        }
        false
    }

    fn add_alias(&mut self, alias: HirId, target: HirId) {
        // This is to prevent alias loop.
        if alias == target || self.would_be_alias_cycle(alias, target) {
            return;
        }
        self.aliases.insert(alias, target);
    }

    // The goal here is to find if the current scope is unsafe or not. It stops when it finds
    // a function or an unsafe block.
    fn is_in_unsafe_block(&self, item: HirId) -> bool {
        for (parent, node) in self.tcx.hir_parent_iter(item) {
            if let Some(fn_sig) = self.tcx.hir_fn_sig_by_hir_id(parent) {
                return fn_sig.header.is_unsafe();
            } else if let Node::Block(block) = node
                && matches!(block.rules, BlockCheckMode::UnsafeBlock(_))
            {
                return true;
            }
        }
        false
    }
}

impl<'tcx> euv::Delegate<'tcx> for MutablyUsedVariablesCtxt<'tcx> {
    #[allow(clippy::if_same_then_else)]
    fn consume(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, id: HirId) {
        if let euv::Place {
            base:
                euv::PlaceBase::Local(vid)
                | euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            base_ty,
            ..
        } = &cmt.place
        {
            if let Some(bind_id) = self.prev_bind.take() {
                if bind_id != *vid {
                    self.add_alias(bind_id, *vid);
                }
            } else if !self.prev_move_to_closure.contains(vid)
                && matches!(base_ty.ref_mutability(), Some(Mutability::Mut))
            {
                self.add_mutably_used_var(*vid);
            } else if self.is_in_unsafe_block(id) {
                // If we are in an unsafe block, any operation on this variable must not be warned
                // upon!
                self.add_mutably_used_var(*vid);
            }
            self.prev_bind = None;
            // FIXME(rust/#120456) - is `swap_remove` correct?
            self.prev_move_to_closure.swap_remove(vid);
        }
    }

    fn use_cloned(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId) {}

    #[allow(clippy::if_same_then_else)]
    fn borrow(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, id: HirId, borrow: ty::BorrowKind) {
        self.prev_bind = None;
        if let euv::Place {
            base:
                euv::PlaceBase::Local(vid)
                | euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            base_ty,
            ..
        } = &cmt.place
        {
            // If this is a mutable borrow, it was obviously used mutably so we add it. However
            // for `UniqueImmBorrow`, it's interesting because if you do: `array[0] = value` inside
            // a closure, it'll return this variant whereas if you have just an index access, it'll
            // return `ImmBorrow`. So if there is "Unique" and it's a mutable reference, we add it
            // to the mutably used variables set.
            if borrow == ty::BorrowKind::Mutable
                || (borrow == ty::BorrowKind::UniqueImmutable && base_ty.ref_mutability() == Some(Mutability::Mut))
            {
                self.add_mutably_used_var(*vid);
            } else if self.is_in_unsafe_block(id) {
                // If we are in an unsafe block, any operation on this variable must not be warned
                // upon!
                self.add_mutably_used_var(*vid);
            }
        } else if borrow == ty::BorrowKind::Immutable
            // If there is an `async block`, it'll contain a call to a closure which we need to
            // go into to ensure all "mutate" checks are found.
            && let Node::Expr(Expr {
                kind:
                    ExprKind::Call(
                        _,
                        [
                            Expr {
                                kind: ExprKind::Closure(Closure { def_id, .. }),
                                ..
                            },
                        ],
                    ),
                ..
            }) = self.tcx.hir_node(cmt.hir_id)
        {
            self.async_closures.insert(*def_id);
        }
    }

    fn mutate(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, _id: HirId) {
        self.prev_bind = None;
        if let euv::Place {
            projections,
            base:
                euv::PlaceBase::Local(vid)
                | euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            ..
        } = &cmt.place
            && !projections.is_empty()
        {
            self.add_mutably_used_var(*vid);
        }
    }

    fn copy(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, id: HirId) {
        if let euv::Place {
            base:
                euv::PlaceBase::Local(vid)
                | euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            ..
        } = &cmt.place
            && self.is_in_unsafe_block(id)
        {
            self.add_mutably_used_var(*vid);
        }
        self.prev_bind = None;
    }

    fn fake_read(
        &mut self,
        cmt: &rustc_hir_typeck::expr_use_visitor::PlaceWithHirId<'tcx>,
        cause: FakeReadCause,
        _id: HirId,
    ) {
        if let euv::Place {
            base:
                euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            ..
        } = &cmt.place
            && let FakeReadCause::ForLet(Some(inner)) = cause
        {
            // Seems like we are inside an async function. We need to store the closure `DefId`
            // to go through it afterwards.
            self.async_closures.insert(inner);
            self.add_alias(cmt.hir_id, *vid);
            self.prev_move_to_closure.insert(*vid);
            self.prev_bind = None;
        }
    }

    fn bind(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, id: HirId) {
        self.prev_bind = Some(id);
        if let euv::Place {
            base:
                euv::PlaceBase::Local(vid)
                | euv::PlaceBase::Upvar(UpvarId {
                    var_path: UpvarPath { hir_id: vid },
                    ..
                }),
            ..
        } = &cmt.place
            && self.is_in_unsafe_block(id)
        {
            self.add_mutably_used_var(*vid);
        }
    }
}
