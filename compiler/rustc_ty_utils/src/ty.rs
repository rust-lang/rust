use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_index::bit_set::BitSet;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, EarlyBinder, ToPredicate, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits;

fn sized_constraint_for_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    adtdef: ty::AdtDef<'tcx>,
    ty: Ty<'tcx>,
) -> Vec<Ty<'tcx>> {
    use rustc_type_ir::sty::TyKind::*;

    let result = match ty.kind() {
        Bool | Char | Int(..) | Uint(..) | Float(..) | RawPtr(..) | Ref(..) | FnDef(..)
        | FnPtr(_) | Array(..) | Closure(..) | Generator(..) | Never => vec![],

        Str
        | Dynamic(..)
        | Slice(_)
        | Foreign(..)
        | Error(_)
        | GeneratorWitness(..)
        | GeneratorWitnessMIR(..) => {
            // these are never sized - return the target type
            vec![ty]
        }

        Tuple(ref tys) => match tys.last() {
            None => vec![],
            Some(&ty) => sized_constraint_for_ty(tcx, adtdef, ty),
        },

        Adt(adt, substs) => {
            // recursive case
            let adt_tys = adt.sized_constraint(tcx);
            debug!("sized_constraint_for_ty({:?}) intermediate = {:?}", ty, adt_tys);
            adt_tys
                .subst_iter_copied(tcx, substs)
                .flat_map(|ty| sized_constraint_for_ty(tcx, adtdef, ty))
                .collect()
        }

        Alias(..) => {
            // must calculate explicitly.
            // FIXME: consider special-casing always-Sized projections
            vec![ty]
        }

        Param(..) => {
            // perf hack: if there is a `T: Sized` bound, then
            // we know that `T` is Sized and do not need to check
            // it on the impl.

            let Some(sized_trait) = tcx.lang_items().sized_trait() else { return vec![ty] };
            let sized_predicate =
                ty::TraitRef::new(tcx, sized_trait, [ty]).without_const().to_predicate(tcx);
            let predicates = tcx.predicates_of(adtdef.did()).predicates;
            if predicates.iter().any(|(p, _)| *p == sized_predicate) { vec![] } else { vec![ty] }
        }

        Placeholder(..) | Bound(..) | Infer(..) => {
            bug!("unexpected type `{:?}` in sized_constraint_for_ty", ty)
        }
    };
    debug!("sized_constraint_for_ty({:?}) = {:?}", ty, result);
    result
}

fn defaultness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::Defaultness {
    match tcx.hir().get_by_def_id(def_id) {
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
///     - a type parameter or projection whose Sizedness can't be known
///     - a tuple of type parameters or projections, if there are multiple
///       such.
///     - an Error, if a type is infinitely sized
fn adt_sized_constraint(tcx: TyCtxt<'_>, def_id: DefId) -> &[Ty<'_>] {
    if let Some(def_id) = def_id.as_local() {
        if matches!(tcx.representability(def_id), ty::Representability::Infinite) {
            return tcx.mk_type_list(&[Ty::new_misc_error(tcx)]);
        }
    }
    let def = tcx.adt_def(def_id);

    let result = tcx.mk_type_list_from_iter(
        def.variants()
            .iter()
            .filter_map(|v| v.tail_opt())
            .flat_map(|f| sized_constraint_for_ty(tcx, def, tcx.type_of(f.did).subst_identity())),
    );

    debug!("adt_sized_constraint: {:?} => {:?}", def, result);

    result
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
        && tcx.associated_item(def_id).container == ty::AssocItemContainer::TraitContainer
    {
        let sig = tcx.fn_sig(def_id).subst_identity();
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
    // FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty): This isn't correct for
    // RPITITs in const trait fn.
    let hir_id = local_did.and_then(|def_id| tcx.opt_local_def_id_to_hir_id(def_id));

    // FIXME(consts): This is not exactly in line with the constness query.
    let constness = match hir_id {
        Some(hir_id) => match tcx.hir().get(hir_id) {
            hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. })
                if tcx.is_const_default_method(def_id) =>
            {
                hir::Constness::Const
            }

            hir::Node::Item(hir::Item { kind: hir::ItemKind::Const(..), .. })
            | hir::Node::Item(hir::Item { kind: hir::ItemKind::Static(..), .. })
            | hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Const(..), ..
            })
            | hir::Node::AnonConst(_)
            | hir::Node::ConstBlock(_)
            | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. })
            | hir::Node::ImplItem(hir::ImplItem {
                kind:
                    hir::ImplItemKind::Fn(
                        hir::FnSig {
                            header: hir::FnHeader { constness: hir::Constness::Const, .. },
                            ..
                        },
                        ..,
                    ),
                ..
            }) => hir::Constness::Const,

            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Type(..) | hir::ImplItemKind::Fn(..),
                ..
            }) => {
                let parent_hir_id = tcx.hir().parent_id(hir_id);
                match tcx.hir().get(parent_hir_id) {
                    hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { constness, .. }),
                        ..
                    }) => *constness,
                    _ => span_bug!(
                        tcx.def_span(parent_hir_id.owner),
                        "impl item's parent node is not an impl",
                    ),
                }
            }

            hir::Node::Item(hir::Item {
                kind:
                    hir::ItemKind::Fn(hir::FnSig { header: hir::FnHeader { constness, .. }, .. }, ..),
                ..
            })
            | hir::Node::TraitItem(hir::TraitItem {
                kind:
                    hir::TraitItemKind::Fn(
                        hir::FnSig { header: hir::FnHeader { constness, .. }, .. },
                        ..,
                    ),
                ..
            })
            | hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { constness, .. }),
                ..
            }) => *constness,

            _ => hir::Constness::NotConst,
        },
        // FIXME(consts): It's suspicious that a param-env for a foreign item
        // will always have NotConst param-env, though we don't typically use
        // that param-env for anything meaningful right now, so it's likely
        // not an issue.
        None => hir::Constness::NotConst,
    };

    let unnormalized_env =
        ty::ParamEnv::new(tcx.mk_clauses(&predicates), traits::Reveal::UserFacing, constness);

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
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        binder: &ty::Binder<'tcx, T>,
    ) -> std::ops::ControlFlow<Self::BreakTy> {
        self.depth.shift_in(1);
        let binder = binder.super_visit_with(self);
        self.depth.shift_out(1);
        binder
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> std::ops::ControlFlow<Self::BreakTy> {
        if let ty::Alias(ty::Projection, unshifted_alias_ty) = *ty.kind()
            && self.tcx.is_impl_trait_in_trait(unshifted_alias_ty.def_id)
            && self.tcx.impl_trait_in_trait_parent_fn(unshifted_alias_ty.def_id) == self.fn_def_id
            && self.seen.insert(unshifted_alias_ty.def_id)
        {
            // We have entered some binders as we've walked into the
            // bounds of the RPITIT. Shift these binders back out when
            // constructing the top-level projection predicate.
            let shifted_alias_ty = self.tcx.fold_regions(unshifted_alias_ty, |re, depth| {
                if let ty::ReLateBound(index, bv) = re.kind() {
                    if depth != ty::INNERMOST {
                        return ty::Region::new_error_with_message(
                            self.tcx,
                            DUMMY_SP,
                            "we shouldn't walk non-predicate binders with `impl Trait`...",
                        );
                    }
                    ty::Region::new_late_bound(self.tcx, index.shifted_out_to_binder(self.depth), bv)
                } else {
                    re
                }
            });

            // If we're lowering to associated item, install the opaque type which is just
            // the `type_of` of the trait's associated item. If we're using the old lowering
            // strategy, then just reinterpret the associated type like an opaque :^)
            let default_ty = if self.tcx.lower_impl_trait_in_trait_to_assoc_ty() {
                self.tcx.type_of(shifted_alias_ty.def_id).subst(self.tcx, shifted_alias_ty.substs)
            } else {
                Ty::new_alias(self.tcx,ty::Opaque, shifted_alias_ty)
            };

            self.predicates.push(
                ty::Binder::bind_with_vars(
                    ty::ProjectionPredicate { projection_ty: shifted_alias_ty, term: default_ty.into() },
                    self.bound_vars,
                )
                .to_predicate(self.tcx),
            );

            // We walk the *un-shifted* alias ty, because we're tracking the de bruijn
            // binder depth, and if we were to walk `shifted_alias_ty` instead, we'd
            // have to reset `self.depth` back to `ty::INNERMOST` or something. It's
            // easier to just do this.
            for bound in self
                .tcx
                .item_bounds(unshifted_alias_ty.def_id)
                .subst_iter(self.tcx, unshifted_alias_ty.substs)
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

fn instance_def_size_estimate<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_def: ty::InstanceDef<'tcx>,
) -> usize {
    use ty::InstanceDef;

    match instance_def {
        InstanceDef::Item(..) | InstanceDef::DropGlue(..) => {
            let mir = tcx.instance_mir(instance_def);
            mir.basic_blocks.iter().map(|bb| bb.statements.len() + 1).sum()
        }
        // Estimate the size of other compiler-generated shims to be 1.
        _ => 1,
    }
}

/// If `def_id` is an issue 33140 hack impl, returns its self type; otherwise, returns `None`.
///
/// See [`ty::ImplOverlapKind::Issue33140`] for more details.
fn issue33140_self_ty(tcx: TyCtxt<'_>, def_id: DefId) -> Option<EarlyBinder<Ty<'_>>> {
    debug!("issue33140_self_ty({:?})", def_id);

    let trait_ref = tcx
        .impl_trait_ref(def_id)
        .unwrap_or_else(|| bug!("issue33140_self_ty called on inherent impl {:?}", def_id))
        .skip_binder();

    debug!("issue33140_self_ty({:?}), trait-ref={:?}", def_id, trait_ref);

    let is_marker_like = tcx.impl_polarity(def_id) == ty::ImplPolarity::Positive
        && tcx.associated_item_def_ids(trait_ref.def_id).is_empty();

    // Check whether these impls would be ok for a marker trait.
    if !is_marker_like {
        debug!("issue33140_self_ty - not marker-like!");
        return None;
    }

    // impl must be `impl Trait for dyn Marker1 + Marker2 + ...`
    if trait_ref.substs.len() != 1 {
        debug!("issue33140_self_ty - impl has substs!");
        return None;
    }

    let predicates = tcx.predicates_of(def_id);
    if predicates.parent.is_some() || !predicates.predicates.is_empty() {
        debug!("issue33140_self_ty - impl has predicates {:?}!", predicates);
        return None;
    }

    let self_ty = trait_ref.self_ty();
    let self_ty_matches = match self_ty.kind() {
        ty::Dynamic(ref data, re, _) if re.is_static() => data.principal().is_none(),
        _ => false,
    };

    if self_ty_matches {
        debug!("issue33140_self_ty - MATCHES!");
        Some(EarlyBinder::bind(self_ty))
    } else {
        debug!("issue33140_self_ty - non-matching self type");
        None
    }
}

/// Check if a function is async.
fn asyncness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> hir::IsAsync {
    let node = tcx.hir().get_by_def_id(def_id);
    node.fn_sig().map_or(hir::IsAsync::NotAsync, |sig| sig.header.asyncness)
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
    let Some((tail_field, prefix_fields)) =
        def.non_enum_variant().fields.raw.split_last() else
    {
        return BitSet::new_empty(num_params);
    };

    let mut unsizing_params = BitSet::new_empty(num_params);
    for arg in tcx.type_of(tail_field.did).subst_identity().walk() {
        if let Some(i) = maybe_unsizing_param_idx(arg) {
            unsizing_params.insert(i);
        }
    }

    // Ensure none of the other fields mention the parameters used
    // in unsizing.
    for field in prefix_fields {
        for arg in tcx.type_of(field.did).subst_identity().walk() {
            if let Some(i) = maybe_unsizing_param_idx(arg) {
                unsizing_params.remove(i);
            }
        }
    }

    unsizing_params
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        asyncness,
        adt_sized_constraint,
        param_env,
        param_env_reveal_all_normalized,
        instance_def_size_estimate,
        issue33140_self_ty,
        defaultness,
        unsizing_params_for_adt,
        ..*providers
    };
}
