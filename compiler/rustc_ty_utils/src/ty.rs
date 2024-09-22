use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::BitSet;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, EarlyBinder, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor, Upcast,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_trait_selection::traits;
use tracing::{debug, instrument};

#[instrument(level = "debug", skip(tcx), ret)]
fn sized_constraint_for_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    use rustc_type_ir::TyKind::*;

    match ty.kind() {
        // these are always sized
        Bool
        | Char
        | Int(..)
        | Uint(..)
        | Float(..)
        | RawPtr(..)
        | Ref(..)
        | FnDef(..)
        | FnPtr(..)
        | Array(..)
        | Closure(..)
        | CoroutineClosure(..)
        | Coroutine(..)
        | CoroutineWitness(..)
        | Never
        | Dynamic(_, _, ty::DynStar) => None,

        // these are never sized
        Str | Slice(..) | Dynamic(_, _, ty::Dyn) | Foreign(..) => Some(ty),

        Pat(ty, _) => sized_constraint_for_ty(tcx, *ty),

        Tuple(tys) => tys.last().and_then(|&ty| sized_constraint_for_ty(tcx, ty)),

        // recursive case
        Adt(adt, args) => adt.sized_constraint(tcx).and_then(|intermediate| {
            let ty = intermediate.instantiate(tcx, args);
            sized_constraint_for_ty(tcx, ty)
        }),

        // these can be sized or unsized
        Param(..) | Alias(..) | Error(_) => Some(ty),

        Placeholder(..) | Bound(..) | Infer(..) => {
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
    if let Err(guar) = constraint_ty.error_reported() {
        return Some(ty::EarlyBinder::bind(Ty::new_error(tcx, guar)));
    }

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
        && assoc_item.container == ty::AssocItemContainer::TraitContainer
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

    let local_did = def_id.as_local();

    let unnormalized_env =
        ty::ParamEnv::new(tcx.mk_clauses(&predicates), traits::Reveal::UserFacing);

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
            let shifted_alias_ty = self.tcx.fold_regions(unshifted_alias_ty, |re, depth| {
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

fn param_env_reveal_all_normalized(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ParamEnv<'_> {
    tcx.param_env(def_id).with_reveal_all_normalized(tcx)
}

/// If the given trait impl enables exploiting the former order dependence of trait objects,
/// returns its self type; otherwise, returns `None`.
///
/// See [`ty::ImplOverlapKind::FutureCompatOrderDepTraitObjects`] for more details.
#[instrument(level = "debug", skip(tcx))]
fn self_ty_of_trait_impl_enabling_order_dep_trait_object_hack(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Option<EarlyBinder<'_, Ty<'_>>> {
    let impl_ =
        tcx.impl_trait_header(def_id).unwrap_or_else(|| bug!("called on inherent impl {def_id:?}"));

    let trait_ref = impl_.trait_ref.skip_binder();
    debug!(?trait_ref);

    let is_marker_like = impl_.polarity == ty::ImplPolarity::Positive
        && tcx.associated_item_def_ids(trait_ref.def_id).is_empty();

    // Check whether these impls would be ok for a marker trait.
    if !is_marker_like {
        debug!("not marker-like!");
        return None;
    }

    // impl must be `impl Trait for dyn Marker1 + Marker2 + ...`
    if trait_ref.args.len() != 1 {
        debug!("impl has args!");
        return None;
    }

    let predicates = tcx.predicates_of(def_id);
    if predicates.parent.is_some() || !predicates.predicates.is_empty() {
        debug!(?predicates, "impl has predicates!");
        return None;
    }

    let self_ty = trait_ref.self_ty();
    let self_ty_matches = match self_ty.kind() {
        ty::Dynamic(data, re, _) if re.is_static() => data.principal().is_none(),
        _ => false,
    };

    if self_ty_matches {
        debug!("MATCHES!");
        Some(EarlyBinder::bind(self_ty))
    } else {
        debug!("non-matching self type");
        None
    }
}

/// Check if a function is async.
fn asyncness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::Asyncness {
    let node = tcx.hir_node_by_def_id(def_id);
    node.fn_sig().map_or(ty::Asyncness::No, |sig| match sig.header.asyncness {
        hir::IsAsync::Async(_) => ty::Asyncness::Yes,
        hir::IsAsync::NotAsync => ty::Asyncness::No,
    })
}

fn unsizing_params_for_adt<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> BitSet<u32> {
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
        return BitSet::new_empty(num_params);
    };

    let mut unsizing_params = BitSet::new_empty(num_params);
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

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        asyncness,
        adt_sized_constraint,
        param_env,
        param_env_reveal_all_normalized,
        self_ty_of_trait_impl_enabling_order_dep_trait_object_hack,
        defaultness,
        unsizing_params_for_adt,
        ..*providers
    };
}
