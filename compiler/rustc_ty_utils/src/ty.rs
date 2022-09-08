use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Binder, Predicate, PredicateKind, ToPredicate, Ty, TyCtxt};
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

        Str | Dynamic(..) | Slice(_) | Foreign(..) | Error(_) | GeneratorWitness(..) => {
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
                .0
                .iter()
                .map(|ty| adt_tys.rebind(*ty).subst(tcx, substs))
                .flat_map(|ty| sized_constraint_for_ty(tcx, adtdef, ty))
                .collect()
        }

        Projection(..) | Opaque(..) => {
            // must calculate explicitly.
            // FIXME: consider special-casing always-Sized projections
            vec![ty]
        }

        Param(..) => {
            // perf hack: if there is a `T: Sized` bound, then
            // we know that `T` is Sized and do not need to check
            // it on the impl.

            let Some(sized_trait) = tcx.lang_items().sized_trait() else { return vec![ty] };
            let sized_predicate = ty::Binder::dummy(ty::TraitRef {
                def_id: sized_trait,
                substs: tcx.mk_substs_trait(ty, &[]),
            })
            .without_const()
            .to_predicate(tcx);
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

fn impl_defaultness(tcx: TyCtxt<'_>, def_id: DefId) -> hir::Defaultness {
    match tcx.hir().get_by_def_id(def_id.expect_local()) {
        hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(impl_), .. }) => impl_.defaultness,
        hir::Node::ImplItem(hir::ImplItem { defaultness, .. })
        | hir::Node::TraitItem(hir::TraitItem { defaultness, .. }) => *defaultness,
        node => {
            bug!("`impl_defaultness` called on {:?}", node);
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
///     - an Error, if a type contained itself. The representability
///       check should catch this case.
fn adt_sized_constraint(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AdtSizedConstraint<'_> {
    let def = tcx.adt_def(def_id);

    let result = tcx.mk_type_list(
        def.variants()
            .iter()
            .flat_map(|v| v.fields.last())
            .flat_map(|f| sized_constraint_for_ty(tcx, def, tcx.type_of(f.did))),
    );

    debug!("adt_sized_constraint: {:?} => {:?}", def, result);

    ty::AdtSizedConstraint(result)
}

/// See `ParamEnv` struct definition for details.
fn param_env(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ParamEnv<'_> {
    // The param_env of an impl Trait type is its defining function's param_env
    if let Some(parent) = ty::is_impl_trait_defn(tcx, def_id) {
        return param_env(tcx, parent.to_def_id());
    }
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

    if tcx.sess.opts.unstable_opts.chalk {
        let environment = well_formed_types_in_env(tcx, def_id);
        predicates.extend(environment);
    }

    let local_did = def_id.as_local();
    let hir_id = local_did.map(|def_id| tcx.hir().local_def_id_to_hir_id(def_id));

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
                kind: hir::ImplItemKind::TyAlias(..) | hir::ImplItemKind::Fn(..),
                ..
            }) => {
                let parent_hir_id = tcx.hir().get_parent_node(hir_id);
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
        None => hir::Constness::NotConst,
    };

    let unnormalized_env = ty::ParamEnv::new(
        tcx.intern_predicates(&predicates),
        traits::Reveal::UserFacing,
        constness,
    );

    let body_id =
        local_did.and_then(|id| tcx.hir().maybe_body_owned_by(id).map(|body| body.hir_id));
    let body_id = match body_id {
        Some(id) => id,
        None if hir_id.is_some() => hir_id.unwrap(),
        _ => hir::CRATE_HIR_ID,
    };

    let cause = traits::ObligationCause::misc(tcx.def_span(def_id), body_id);
    traits::normalize_param_env_or_error(tcx, unnormalized_env, cause)
}

/// Elaborate the environment.
///
/// Collect a list of `Predicate`'s used for building the `ParamEnv`. Adds `TypeWellFormedFromEnv`'s
/// that are assumed to be well-formed (because they come from the environment).
///
/// Used only in chalk mode.
fn well_formed_types_in_env<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> &'tcx ty::List<Predicate<'tcx>> {
    use rustc_hir::{ForeignItemKind, ImplItemKind, ItemKind, Node, TraitItemKind};
    use rustc_middle::ty::subst::GenericArgKind;

    debug!("environment(def_id = {:?})", def_id);

    // The environment of an impl Trait type is its defining function's environment.
    if let Some(parent) = ty::is_impl_trait_defn(tcx, def_id) {
        return well_formed_types_in_env(tcx, parent.to_def_id());
    }

    // Compute the bounds on `Self` and the type parameters.
    let ty::InstantiatedPredicates { predicates, .. } =
        tcx.predicates_of(def_id).instantiate_identity(tcx);

    let clauses = predicates.into_iter();

    if !def_id.is_local() {
        return ty::List::empty();
    }
    let node = tcx.hir().get_by_def_id(def_id.expect_local());

    enum NodeKind {
        TraitImpl,
        InherentImpl,
        Fn,
        Other,
    }

    let node_kind = match node {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::ImplItem(item) => match item.kind {
            ImplItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::Item(item) => match item.kind {
            ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => NodeKind::TraitImpl,
            ItemKind::Impl(hir::Impl { of_trait: None, .. }) => NodeKind::InherentImpl,
            ItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        // FIXME: closures?
        _ => NodeKind::Other,
    };

    // FIXME(eddyb) isn't the unordered nature of this a hazard?
    let mut inputs = FxIndexSet::default();

    match node_kind {
        // In a trait impl, we assume that the header trait ref and all its
        // constituents are well-formed.
        NodeKind::TraitImpl => {
            let trait_ref = tcx.impl_trait_ref(def_id).expect("not an impl");

            // FIXME(chalk): this has problems because of late-bound regions
            //inputs.extend(trait_ref.substs.iter().flat_map(|arg| arg.walk()));
            inputs.extend(trait_ref.substs.iter());
        }

        // In an inherent impl, we assume that the receiver type and all its
        // constituents are well-formed.
        NodeKind::InherentImpl => {
            let self_ty = tcx.type_of(def_id);
            inputs.extend(self_ty.walk());
        }

        // In an fn, we assume that the arguments and all their constituents are
        // well-formed.
        NodeKind::Fn => {
            let fn_sig = tcx.fn_sig(def_id);
            let fn_sig = tcx.liberate_late_bound_regions(def_id, fn_sig);

            inputs.extend(fn_sig.inputs().iter().flat_map(|ty| ty.walk()));
        }

        NodeKind::Other => (),
    }
    let input_clauses = inputs.into_iter().filter_map(|arg| {
        match arg.unpack() {
            GenericArgKind::Type(ty) => {
                let binder = Binder::dummy(PredicateKind::TypeWellFormedFromEnv(ty));
                Some(tcx.mk_predicate(binder))
            }

            // FIXME(eddyb) no WF conditions from lifetimes?
            GenericArgKind::Lifetime(_) => None,

            // FIXME(eddyb) support const generics in Chalk
            GenericArgKind::Const(_) => None,
        }
    });

    tcx.mk_predicates(clauses.chain(input_clauses))
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
fn issue33140_self_ty(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Ty<'_>> {
    debug!("issue33140_self_ty({:?})", def_id);

    let trait_ref = tcx
        .impl_trait_ref(def_id)
        .unwrap_or_else(|| bug!("issue33140_self_ty called on inherent impl {:?}", def_id));

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
        ty::Dynamic(ref data, re) if re.is_static() => data.principal().is_none(),
        _ => false,
    };

    if self_ty_matches {
        debug!("issue33140_self_ty - MATCHES!");
        Some(self_ty)
    } else {
        debug!("issue33140_self_ty - non-matching self type");
        None
    }
}

/// Check if a function is async.
fn asyncness(tcx: TyCtxt<'_>, def_id: DefId) -> hir::IsAsync {
    let node = tcx.hir().get_by_def_id(def_id.expect_local());
    if let Some(fn_kind) = node.fn_kind() { fn_kind.asyncness() } else { hir::IsAsync::NotAsync }
}

/// Don't call this directly: use ``tcx.conservative_is_privately_uninhabited`` instead.
pub fn conservative_is_privately_uninhabited_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env_and: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> bool {
    let (param_env, ty) = param_env_and.into_parts();
    match ty.kind() {
        ty::Never => {
            debug!("ty::Never =>");
            true
        }
        ty::Adt(def, _) if def.is_union() => {
            debug!("ty::Adt(def, _) if def.is_union() =>");
            // For now, `union`s are never considered uninhabited.
            false
        }
        ty::Adt(def, substs) => {
            debug!("ty::Adt(def, _) if def.is_not_union() =>");
            // Any ADT is uninhabited if either:
            // (a) It has no variants (i.e. an empty `enum`);
            // (b) Each of its variants (a single one in the case of a `struct`) has at least
            //     one uninhabited field.
            def.variants().iter().all(|var| {
                var.fields.iter().any(|field| {
                    let ty = tcx.bound_type_of(field.did).subst(tcx, substs);
                    tcx.conservative_is_privately_uninhabited(param_env.and(ty))
                })
            })
        }
        ty::Tuple(fields) => {
            debug!("ty::Tuple(..) =>");
            fields.iter().any(|ty| tcx.conservative_is_privately_uninhabited(param_env.and(ty)))
        }
        ty::Array(ty, len) => {
            debug!("ty::Array(ty, len) =>");
            match len.try_eval_usize(tcx, param_env) {
                Some(0) | None => false,
                // If the array is definitely non-empty, it's uninhabited if
                // the type of its elements is uninhabited.
                Some(1..) => tcx.conservative_is_privately_uninhabited(param_env.and(*ty)),
            }
        }
        ty::Ref(..) => {
            debug!("ty::Ref(..) =>");
            // References to uninitialised memory is valid for any type, including
            // uninhabited types, in unsafe code, so we treat all references as
            // inhabited.
            false
        }
        _ => {
            debug!("_ =>");
            false
        }
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        asyncness,
        adt_sized_constraint,
        param_env,
        param_env_reveal_all_normalized,
        instance_def_size_estimate,
        issue33140_self_ty,
        impl_defaultness,
        conservative_is_privately_uninhabited: conservative_is_privately_uninhabited_raw,
        ..*providers
    };
}
