use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::svh::Svh;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_middle::hir::map as hir_map;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{
    self, Binder, Predicate, PredicateAtom, PredicateKind, ToPredicate, Ty, TyCtxt, WithConstness,
};
use rustc_session::CrateDisambiguator;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use rustc_trait_selection::traits;

fn sized_constraint_for_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    adtdef: &ty::AdtDef,
    ty: Ty<'tcx>,
) -> Vec<Ty<'tcx>> {
    use ty::TyKind::*;

    let result = match ty.kind() {
        Bool | Char | Int(..) | Uint(..) | Float(..) | RawPtr(..) | Ref(..) | FnDef(..)
        | FnPtr(_) | Array(..) | Closure(..) | Generator(..) | Never => vec![],

        Str | Dynamic(..) | Slice(_) | Foreign(..) | Error(_) | GeneratorWitness(..) => {
            // these are never sized - return the target type
            vec![ty]
        }

        Tuple(ref tys) => match tys.last() {
            None => vec![],
            Some(ty) => sized_constraint_for_ty(tcx, adtdef, ty.expect_ty()),
        },

        Adt(adt, substs) => {
            // recursive case
            let adt_tys = adt.sized_constraint(tcx);
            debug!("sized_constraint_for_ty({:?}) intermediate = {:?}", ty, adt_tys);
            adt_tys
                .iter()
                .map(|ty| ty.subst(tcx, substs))
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

            let sized_trait = match tcx.lang_items().sized_trait() {
                Some(x) => x,
                _ => return vec![ty],
            };
            let sized_predicate = ty::Binder::dummy(ty::TraitRef {
                def_id: sized_trait,
                substs: tcx.mk_substs_trait(ty, &[]),
            })
            .without_const()
            .to_predicate(tcx);
            let predicates = tcx.predicates_of(adtdef.did).predicates;
            if predicates.iter().any(|(p, _)| *p == sized_predicate) { vec![] } else { vec![ty] }
        }

        Placeholder(..) | Bound(..) | Infer(..) => {
            bug!("unexpected type `{:?}` in sized_constraint_for_ty", ty)
        }
    };
    debug!("sized_constraint_for_ty({:?}) = {:?}", ty, result);
    result
}

fn associated_item_from_trait_item_ref(
    tcx: TyCtxt<'_>,
    parent_def_id: LocalDefId,
    trait_item_ref: &hir::TraitItemRef,
) -> ty::AssocItem {
    let def_id = tcx.hir().local_def_id(trait_item_ref.id.hir_id);
    let (kind, has_self) = match trait_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        ident: trait_item_ref.ident,
        kind,
        vis: tcx.visibility(def_id),
        defaultness: trait_item_ref.defaultness,
        def_id: def_id.to_def_id(),
        container: ty::TraitContainer(parent_def_id.to_def_id()),
        fn_has_self_parameter: has_self,
    }
}

fn associated_item_from_impl_item_ref(
    tcx: TyCtxt<'_>,
    parent_def_id: LocalDefId,
    impl_item_ref: &hir::ImplItemRef<'_>,
) -> ty::AssocItem {
    let def_id = tcx.hir().local_def_id(impl_item_ref.id.hir_id);
    let (kind, has_self) = match impl_item_ref.kind {
        hir::AssocItemKind::Const => (ty::AssocKind::Const, false),
        hir::AssocItemKind::Fn { has_self } => (ty::AssocKind::Fn, has_self),
        hir::AssocItemKind::Type => (ty::AssocKind::Type, false),
    };

    ty::AssocItem {
        ident: impl_item_ref.ident,
        kind,
        vis: tcx.visibility(def_id),
        defaultness: impl_item_ref.defaultness,
        def_id: def_id.to_def_id(),
        container: ty::ImplContainer(parent_def_id.to_def_id()),
        fn_has_self_parameter: has_self,
    }
}

fn associated_item(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssocItem {
    let id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let parent_id = tcx.hir().get_parent_item(id);
    let parent_def_id = tcx.hir().local_def_id(parent_id);
    let parent_item = tcx.hir().expect_item(parent_id);
    match parent_item.kind {
        hir::ItemKind::Impl { ref items, .. } => {
            if let Some(impl_item_ref) = items.iter().find(|i| i.id.hir_id == id) {
                let assoc_item =
                    associated_item_from_impl_item_ref(tcx, parent_def_id, impl_item_ref);
                debug_assert_eq!(assoc_item.def_id, def_id);
                return assoc_item;
            }
        }

        hir::ItemKind::Trait(.., ref trait_item_refs) => {
            if let Some(trait_item_ref) = trait_item_refs.iter().find(|i| i.id.hir_id == id) {
                let assoc_item =
                    associated_item_from_trait_item_ref(tcx, parent_def_id, trait_item_ref);
                debug_assert_eq!(assoc_item.def_id, def_id);
                return assoc_item;
            }
        }

        _ => {}
    }

    span_bug!(
        parent_item.span,
        "unexpected parent of trait or impl item or item not found: {:?}",
        parent_item.kind
    )
}

fn impl_defaultness(tcx: TyCtxt<'_>, def_id: DefId) -> hir::Defaultness {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let item = tcx.hir().expect_item(hir_id);
    if let hir::ItemKind::Impl { defaultness, .. } = item.kind {
        defaultness
    } else {
        bug!("`impl_defaultness` called on {:?}", item);
    }
}

/// Calculates the `Sized` constraint.
///
/// In fact, there are only a few options for the types in the constraint:
///     - an obviously-unsized type
///     - a type parameter or projection whose Sizedness can't be known
///     - a tuple of type parameters or projections, if there are multiple
///       such.
///     - a Error, if a type contained itself. The representability
///       check should catch this case.
fn adt_sized_constraint(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AdtSizedConstraint<'_> {
    let def = tcx.adt_def(def_id);

    let result = tcx.mk_type_list(
        def.variants
            .iter()
            .flat_map(|v| v.fields.last())
            .flat_map(|f| sized_constraint_for_ty(tcx, def, tcx.type_of(f.did))),
    );

    debug!("adt_sized_constraint: {:?} => {:?}", def, result);

    ty::AdtSizedConstraint(result)
}

fn associated_item_def_ids(tcx: TyCtxt<'_>, def_id: DefId) -> &[DefId] {
    let id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let item = tcx.hir().expect_item(id);
    match item.kind {
        hir::ItemKind::Trait(.., ref trait_item_refs) => tcx.arena.alloc_from_iter(
            trait_item_refs
                .iter()
                .map(|trait_item_ref| trait_item_ref.id)
                .map(|id| tcx.hir().local_def_id(id.hir_id).to_def_id()),
        ),
        hir::ItemKind::Impl { ref items, .. } => tcx.arena.alloc_from_iter(
            items
                .iter()
                .map(|impl_item_ref| impl_item_ref.id)
                .map(|id| tcx.hir().local_def_id(id.hir_id).to_def_id()),
        ),
        hir::ItemKind::TraitAlias(..) => &[],
        _ => span_bug!(item.span, "associated_item_def_ids: not impl or trait"),
    }
}

fn associated_items(tcx: TyCtxt<'_>, def_id: DefId) -> ty::AssociatedItems<'_> {
    let items = tcx.associated_item_def_ids(def_id).iter().map(|did| tcx.associated_item(*did));
    ty::AssociatedItems::new(items)
}

fn def_span(tcx: TyCtxt<'_>, def_id: DefId) -> Span {
    tcx.hir().span_if_local(def_id).unwrap()
}

/// If the given `DefId` describes an item belonging to a trait,
/// returns the `DefId` of the trait that the trait item belongs to;
/// otherwise, returns `None`.
fn trait_of_item(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    tcx.opt_associated_item(def_id).and_then(|associated_item| match associated_item.container {
        ty::TraitContainer(def_id) => Some(def_id),
        ty::ImplContainer(_) => None,
    })
}

/// See `ParamEnv` struct definition for details.
fn param_env(tcx: TyCtxt<'_>, def_id: DefId) -> ty::ParamEnv<'_> {
    // The param_env of an impl Trait type is its defining function's param_env
    if let Some(parent) = ty::is_impl_trait_defn(tcx, def_id) {
        return param_env(tcx, parent);
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
    // kind of a "idempotent" action, but I'm not sure where would be
    // a better place. In practice, we construct environments for
    // every fn once during type checking, and we'll abort if there
    // are any errors at that point, so after type checking you can be
    // sure that this will succeed without errors anyway.

    if tcx.sess.opts.debugging_opts.chalk {
        let environment = well_formed_types_in_env(tcx, def_id);
        predicates.extend(environment);
    }

    let unnormalized_env =
        ty::ParamEnv::new(tcx.intern_predicates(&predicates), traits::Reveal::UserFacing);

    let body_id = def_id
        .as_local()
        .map(|def_id| tcx.hir().local_def_id_to_hir_id(def_id))
        .map_or(hir::CRATE_HIR_ID, |id| {
            tcx.hir().maybe_body_owned_by(id).map_or(id, |body| body.hir_id)
        });
    let cause = traits::ObligationCause::misc(tcx.def_span(def_id), body_id);
    traits::normalize_param_env_or_error(tcx, def_id, unnormalized_env, cause)
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
        return well_formed_types_in_env(tcx, parent);
    }

    // Compute the bounds on `Self` and the type parameters.
    let ty::InstantiatedPredicates { predicates, .. } =
        tcx.predicates_of(def_id).instantiate_identity(tcx);

    let clauses = predicates.into_iter();

    if !def_id.is_local() {
        return ty::List::empty();
    }
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
    let node = tcx.hir().get(hir_id);

    enum NodeKind {
        TraitImpl,
        InherentImpl,
        Fn,
        Other,
    };

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
            ItemKind::Impl { of_trait: Some(_), .. } => NodeKind::TraitImpl,
            ItemKind::Impl { of_trait: None, .. } => NodeKind::InherentImpl,
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
                let binder = Binder::dummy(PredicateAtom::TypeWellFormedFromEnv(ty));
                Some(tcx.mk_predicate(PredicateKind::ForAll(binder)))
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

fn crate_disambiguator(tcx: TyCtxt<'_>, crate_num: CrateNum) -> CrateDisambiguator {
    assert_eq!(crate_num, LOCAL_CRATE);
    tcx.sess.local_crate_disambiguator()
}

fn original_crate_name(tcx: TyCtxt<'_>, crate_num: CrateNum) -> Symbol {
    assert_eq!(crate_num, LOCAL_CRATE);
    tcx.crate_name
}

fn crate_hash(tcx: TyCtxt<'_>, crate_num: CrateNum) -> Svh {
    tcx.index_hir(crate_num).crate_hash
}

fn instance_def_size_estimate<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_def: ty::InstanceDef<'tcx>,
) -> usize {
    use ty::InstanceDef;

    match instance_def {
        InstanceDef::Item(..) | InstanceDef::DropGlue(..) => {
            let mir = tcx.instance_mir(instance_def);
            mir.basic_blocks().iter().map(|bb| bb.statements.len()).sum()
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
        ty::Dynamic(ref data, ty::ReStatic) => data.principal().is_none(),
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
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());

    let node = tcx.hir().get(hir_id);

    let fn_like = hir_map::blocks::FnLikeNode::from_node(node).unwrap_or_else(|| {
        bug!("asyncness: expected fn-like node but got `{:?}`", def_id);
    });

    fn_like.asyncness()
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        asyncness,
        associated_item,
        associated_item_def_ids,
        associated_items,
        adt_sized_constraint,
        def_span,
        param_env,
        param_env_reveal_all_normalized,
        trait_of_item,
        crate_disambiguator,
        original_crate_name,
        crate_hash,
        instance_def_size_estimate,
        issue33140_self_ty,
        impl_defaultness,
        ..*providers
    };
}
