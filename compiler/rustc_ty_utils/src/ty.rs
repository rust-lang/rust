use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor, Upcast, fold_regions,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_trait_selection::traits;
use tracing::instrument;

#[instrument(level = "debug", skip(tcx), ret)]
fn sized_constraint_for_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.kind() {
        // these are always sized
        ty::Bool
        | ty::Char
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::FnDef(..)
        | ty::FnPtr(..)
        | ty::Array(..)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::Never
        | ty::Dynamic(_, _, ty::DynStar) => None,

        // these are never sized
        ty::Str | ty::Slice(..) | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => Some(ty),

        ty::Pat(ty, _) => sized_constraint_for_ty(tcx, *ty),

        ty::Tuple(tys) => tys.last().and_then(|&ty| sized_constraint_for_ty(tcx, ty)),

        // recursive case
        ty::Adt(adt, args) => adt.sized_constraint(tcx).and_then(|intermediate| {
            let ty = intermediate.instantiate(tcx, args);
            sized_constraint_for_ty(tcx, ty)
        }),

        // these can be sized or unsized.
        ty::Param(..) | ty::Alias(..) | ty::Error(_) => Some(ty),

        // We cannot instantiate the binder, so just return the *original* type back,
        // but only if the inner type has a sized constraint. Thus we skip the binder,
        // but don't actually use the result from `sized_constraint_for_ty`.
        ty::UnsafeBinder(inner_ty) => {
            sized_constraint_for_ty(tcx, inner_ty.skip_binder()).map(|_| ty)
        }

        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) => {
            bug!("unexpected type `{ty:?}` in sized_constraint_for_ty")
        }
    }
}

fn defaultness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::Defaultness {
    match tcx.hir_node_by_def_id(def_id) {
        hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(impl_), .. }) => impl_.defaultness,
        hir::Node::ImplItem(hir::ImplItem { defaultness, .. })
        | hir::Node::TraitItem(hir::TraitItem { defaultness, .. }) => *defaultness,
        node => {
            bug!("`defaultness` called on {:?}", node);
        }
    }
}

/// Calculates the `Sized` constraint.
///
/// In fact, there are only a few options for the types in the constraint:
///     - an obviously-unsized type
///     - a type parameter or projection whose sizedness can't be known
#[instrument(level = "debug", skip(tcx), ret)]
fn adt_sized_constraint<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Option<ty::EarlyBinder<'tcx, Ty<'tcx>>> {
    if let Some(def_id) = def_id.as_local() {
        if let ty::Representability::Infinite(_) = tcx.representability(def_id) {
            return None;
        }
    }
    let def = tcx.adt_def(def_id);

    if !def.is_struct() {
        bug!("`adt_sized_constraint` called on non-struct type: {def:?}");
    }

    let tail_def = def.non_enum_variant().tail_opt()?;
    let tail_ty = tcx.type_of(tail_def.did).instantiate_identity();

    let constraint_ty = sized_constraint_for_ty(tcx, tail_ty)?;

    // perf hack: if there is a `constraint_ty: Sized` bound, then we know
    // that the type is sized and do not need to check it on the impl.
    let sized_trait_def_id = tcx.require_lang_item(LangItem::Sized, None);
    let predicates = tcx.predicates_of(def.did()).predicates;
    if predicates.iter().any(|(p, _)| {
        p.as_trait_clause().is_some_and(|trait_pred| {
            trait_pred.def_id() == sized_trait_def_id
                && trait_pred.self_ty().skip_binder() == constraint_ty
        })
    }) {
        return None;
    }

    Some(ty::EarlyBinder::bind(constraint_ty))
}

/// See `ParamEnv` struct definition for details.
fn param_env(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ParamEnv<'_> {
    // Compute the bounds on Self and the type parameters.
    let ty::InstantiatedPredicates { mut predicates, .. } =
        tcx.predicates_of(def_id).instantiate_identity(tcx);

    // Finally, we have to normalize the bounds in the environment, in
    // case they contain any associated type projections. This process
    // can yield errors if the put in illegal associated types, like
    // `<i32 as Foo>::Bar` where `i32` does not implement `Foo`. We
    // report these errors right here; this doesn't actually feel
    // right to me, because constructing the environment feels like a
    // kind of an "idempotent" action, but I'm not sure where would be
    // a better place. In practice, we construct environments for
    // every fn once during type checking, and we'll abort if there
    // are any errors at that point, so outside of type inference you can be
    // sure that this will succeed without errors anyway.

    if tcx.def_kind(def_id) == DefKind::AssocFn
        && let assoc_item = tcx.associated_item(def_id)
        && assoc_item.container == ty::AssocItemContainer::Trait
        && assoc_item.defaultness(tcx).has_value()
    {
        let sig = tcx.fn_sig(def_id).instantiate_identity();
        // We accounted for the binder of the fn sig, so skip the binder.
        sig.skip_binder().visit_with(&mut ImplTraitInTraitFinder {
            tcx,
            fn_def_id: def_id,
            bound_vars: sig.bound_vars(),
            predicates: &mut predicates,
            seen: FxHashSet::default(),
            depth: ty::INNERMOST,
        });
    }

    // We extend the param-env of our item with the const conditions of the item,
    // since we're allowed to assume `~const` bounds hold within the item itself.
    if tcx.is_conditionally_const(def_id) {
        predicates.extend(
            tcx.const_conditions(def_id).instantiate_identity(tcx).into_iter().map(
                |(trait_ref, _)| trait_ref.to_host_effect_clause(tcx, ty::BoundConstness::Maybe),
            ),
        );
    }

    let local_did = def_id.as_local();

    let unnormalized_env = ty::ParamEnv::new(tcx.mk_clauses(&predicates));

    let body_id = local_did.unwrap_or(CRATE_DEF_ID);
    let cause = traits::ObligationCause::misc(tcx.def_span(def_id), body_id);
    traits::normalize_param_env_or_error(tcx, unnormalized_env, cause)
}

/// Walk through a function type, gathering all RPITITs and installing a
/// `NormalizesTo(Projection(RPITIT) -> Opaque(RPITIT))` predicate into the
/// predicates list. This allows us to observe that an RPITIT projects to
/// its corresponding opaque within the body of a default-body trait method.
struct ImplTraitInTraitFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    predicates: &'a mut Vec<ty::Clause<'tcx>>,
    fn_def_id: DefId,
    bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
    seen: FxHashSet<DefId>,
    depth: ty::DebruijnIndex,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ImplTraitInTraitFinder<'_, 'tcx> {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(&mut self, binder: &ty::Binder<'tcx, T>) {
        self.depth.shift_in(1);
        binder.super_visit_with(self);
        self.depth.shift_out(1);
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if let ty::Alias(ty::Projection, unshifted_alias_ty) = *ty.kind()
            && let Some(
                ty::ImplTraitInTraitData::Trait { fn_def_id, .. }
                | ty::ImplTraitInTraitData::Impl { fn_def_id, .. },
            ) = self.tcx.opt_rpitit_info(unshifted_alias_ty.def_id)
            && fn_def_id == self.fn_def_id
            && self.seen.insert(unshifted_alias_ty.def_id)
        {
            // We have entered some binders as we've walked into the
            // bounds of the RPITIT. Shift these binders back out when
            // constructing the top-level projection predicate.
            let shifted_alias_ty = fold_regions(self.tcx, unshifted_alias_ty, |re, depth| {
                if let ty::ReBound(index, bv) = re.kind() {
                    if depth != ty::INNERMOST {
                        return ty::Region::new_error_with_message(
                            self.tcx,
                            DUMMY_SP,
                            "we shouldn't walk non-predicate binders with `impl Trait`...",
                        );
                    }
                    ty::Region::new_bound(self.tcx, index.shifted_out_to_binder(self.depth), bv)
                } else {
                    re
                }
            });

            // If we're lowering to associated item, install the opaque type which is just
            // the `type_of` of the trait's associated item. If we're using the old lowering
            // strategy, then just reinterpret the associated type like an opaque :^)
            let default_ty = self
                .tcx
                .type_of(shifted_alias_ty.def_id)
                .instantiate(self.tcx, shifted_alias_ty.args);

            self.predicates.push(
                ty::Binder::bind_with_vars(
                    ty::ProjectionPredicate {
                        projection_term: shifted_alias_ty.into(),
                        term: default_ty.into(),
                    },
                    self.bound_vars,
                )
                .upcast(self.tcx),
            );

            // We walk the *un-shifted* alias ty, because we're tracking the de bruijn
            // binder depth, and if we were to walk `shifted_alias_ty` instead, we'd
            // have to reset `self.depth` back to `ty::INNERMOST` or something. It's
            // easier to just do this.
            for bound in self
                .tcx
                .item_bounds(unshifted_alias_ty.def_id)
                .iter_instantiated(self.tcx, unshifted_alias_ty.args)
            {
                bound.visit_with(self);
            }
        }

        ty.super_visit_with(self)
    }
}

fn param_env_normalized_for_post_analysis(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ParamEnv<'_> {
    // This is a bit ugly but the easiest way to avoid code duplication.
    let typing_env = ty::TypingEnv::non_body_analysis(tcx, def_id);
    typing_env.with_post_analysis_normalized(tcx).param_env
}

/// Check if a function is async.
fn asyncness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::Asyncness {
    let node = tcx.hir_node_by_def_id(def_id);
    node.fn_sig().map_or(ty::Asyncness::No, |sig| match sig.header.asyncness {
        hir::IsAsync::Async(_) => ty::Asyncness::Yes,
        hir::IsAsync::NotAsync => ty::Asyncness::No,
    })
}

fn unsizing_params_for_adt<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> DenseBitSet<u32> {
    let def = tcx.adt_def(def_id);
    let num_params = tcx.generics_of(def_id).count();

    let maybe_unsizing_param_idx = |arg: ty::GenericArg<'tcx>| match arg.unpack() {
        ty::GenericArgKind::Type(ty) => match ty.kind() {
            ty::Param(p) => Some(p.index),
            _ => None,
        },

        // We can't unsize a lifetime
        ty::GenericArgKind::Lifetime(_) => None,

        ty::GenericArgKind::Const(ct) => match ct.kind() {
            ty::ConstKind::Param(p) => Some(p.index),
            _ => None,
        },
    };

    // The last field of the structure has to exist and contain type/const parameters.
    let Some((tail_field, prefix_fields)) = def.non_enum_variant().fields.raw.split_last() else {
        return DenseBitSet::new_empty(num_params);
    };

    let mut unsizing_params = DenseBitSet::new_empty(num_params);
    for arg in tcx.type_of(tail_field.did).instantiate_identity().walk() {
        if let Some(i) = maybe_unsizing_param_idx(arg) {
            unsizing_params.insert(i);
        }
    }

    // Ensure none of the other fields mention the parameters used
    // in unsizing.
    for field in prefix_fields {
        for arg in tcx.type_of(field.did).instantiate_identity().walk() {
            if let Some(i) = maybe_unsizing_param_idx(arg) {
                unsizing_params.remove(i);
            }
        }
    }

    unsizing_params
}

fn impl_self_is_guaranteed_unsized<'tcx>(tcx: TyCtxt<'tcx>, impl_def_id: DefId) -> bool {
    debug_assert_eq!(tcx.def_kind(impl_def_id), DefKind::Impl { of_trait: true });

    let infcx = tcx.infer_ctxt().ignoring_regions().build(ty::TypingMode::non_body_analysis());

    let ocx = traits::ObligationCtxt::new_with_diagnostics(&infcx);
    let cause = traits::ObligationCause::dummy();
    let param_env = tcx.param_env(impl_def_id);

    let tail = tcx.struct_tail_raw(
        tcx.type_of(impl_def_id).instantiate_identity(),
        |ty| {
            ocx.structurally_normalize_ty(&cause, param_env, ty).unwrap_or_else(|_| {
                Ty::new_error_with_message(
                    tcx,
                    tcx.def_span(impl_def_id),
                    "struct tail should be computable",
                )
            })
        },
        || (),
    );

    match tail.kind() {
        ty::Dynamic(_, _, ty::Dyn) | ty::Slice(_) | ty::Str => true,
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Adt(_, _)
        | ty::Foreign(_)
        | ty::Array(_, _)
        | ty::Pat(_, _)
        | ty::RawPtr(_, _)
        | ty::Ref(_, _, _)
        | ty::FnDef(_, _)
        | ty::FnPtr(_, _)
        | ty::UnsafeBinder(_)
        | ty::Closure(_, _)
        | ty::CoroutineClosure(_, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(_, _)
        | ty::Never
        | ty::Tuple(_)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Bound(_, _)
        | ty::Placeholder(_)
        | ty::Infer(_)
        | ty::Error(_)
        | ty::Dynamic(_, _, ty::DynStar) => false,
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        asyncness,
        adt_sized_constraint,
        param_env,
        param_env_normalized_for_post_analysis,
        defaultness,
        unsizing_params_for_adt,
        impl_self_is_guaranteed_unsized,
        ..*providers
    };
}
