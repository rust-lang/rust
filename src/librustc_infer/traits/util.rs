use rustc_errors::DiagnosticBuilder;
use rustc_span::Span;
use smallvec::smallvec;
use smallvec::SmallVec;

use rustc::ty::outlives::Component;
use rustc::ty::subst::{GenericArg, Subst, SubstsRef};
use rustc::ty::{self, ToPolyTraitRef, ToPredicate, Ty, TyCtxt, WithConstness};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;

use super::{Normalized, Obligation, ObligationCause, PredicateObligation, SelectionContext};

fn anonymize_predicate<'tcx>(tcx: TyCtxt<'tcx>, pred: &ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
    match *pred {
        ty::Predicate::Trait(ref data, constness) => {
            ty::Predicate::Trait(tcx.anonymize_late_bound_regions(data), constness)
        }

        ty::Predicate::RegionOutlives(ref data) => {
            ty::Predicate::RegionOutlives(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::TypeOutlives(ref data) => {
            ty::Predicate::TypeOutlives(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::Projection(ref data) => {
            ty::Predicate::Projection(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::WellFormed(data) => ty::Predicate::WellFormed(data),

        ty::Predicate::ObjectSafe(data) => ty::Predicate::ObjectSafe(data),

        ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
            ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind)
        }

        ty::Predicate::Subtype(ref data) => {
            ty::Predicate::Subtype(tcx.anonymize_late_bound_regions(data))
        }

        ty::Predicate::ConstEvaluatable(def_id, substs) => {
            ty::Predicate::ConstEvaluatable(def_id, substs)
        }
    }
}

struct PredicateSet<'tcx> {
    tcx: TyCtxt<'tcx>,
    set: FxHashSet<ty::Predicate<'tcx>>,
}

impl PredicateSet<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, set: Default::default() }
    }

    fn insert(&mut self, pred: &ty::Predicate<'tcx>) -> bool {
        // We have to be careful here because we want
        //
        //    for<'a> Foo<&'a int>
        //
        // and
        //
        //    for<'b> Foo<&'b int>
        //
        // to be considered equivalent. So normalize all late-bound
        // regions before we throw things into the underlying set.
        self.set.insert(anonymize_predicate(self.tcx, pred))
    }
}

impl<T: AsRef<ty::Predicate<'tcx>>> Extend<T> for PredicateSet<'tcx> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for pred in iter {
            self.insert(pred.as_ref());
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// `Elaboration` iterator
///////////////////////////////////////////////////////////////////////////

/// "Elaboration" is the process of identifying all the predicates that
/// are implied by a source predicate. Currently, this basically means
/// walking the "supertraits" and other similar assumptions. For example,
/// if we know that `T: Ord`, the elaborator would deduce that `T: PartialOrd`
/// holds as well. Similarly, if we have `trait Foo: 'static`, and we know that
/// `T: Foo`, then we know that `T: 'static`.
pub struct Elaborator<'tcx> {
    stack: Vec<ty::Predicate<'tcx>>,
    visited: PredicateSet<'tcx>,
}

pub fn elaborate_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> Elaborator<'tcx> {
    elaborate_predicates(tcx, vec![trait_ref.without_const().to_predicate()])
}

pub fn elaborate_trait_refs<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_refs: impl Iterator<Item = ty::PolyTraitRef<'tcx>>,
) -> Elaborator<'tcx> {
    let predicates = trait_refs.map(|trait_ref| trait_ref.without_const().to_predicate()).collect();
    elaborate_predicates(tcx, predicates)
}

pub fn elaborate_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut predicates: Vec<ty::Predicate<'tcx>>,
) -> Elaborator<'tcx> {
    let mut visited = PredicateSet::new(tcx);
    predicates.retain(|pred| visited.insert(pred));
    Elaborator { stack: predicates, visited }
}

impl Elaborator<'tcx> {
    pub fn filter_to_traits(self) -> FilterToTraits<Self> {
        FilterToTraits::new(self)
    }

    fn elaborate(&mut self, predicate: &ty::Predicate<'tcx>) {
        let tcx = self.visited.tcx;
        match *predicate {
            ty::Predicate::Trait(ref data, _) => {
                // Get predicates declared on the trait.
                let predicates = tcx.super_predicates_of(data.def_id());

                let predicates = predicates
                    .predicates
                    .iter()
                    .map(|(pred, _)| pred.subst_supertrait(tcx, &data.to_poly_trait_ref()));
                debug!("super_predicates: data={:?} predicates={:?}", data, predicates.clone());

                // Only keep those bounds that we haven't already seen.
                // This is necessary to prevent infinite recursion in some
                // cases. One common case is when people define
                // `trait Sized: Sized { }` rather than `trait Sized { }`.
                let visited = &mut self.visited;
                let predicates = predicates.filter(|pred| visited.insert(pred));

                self.stack.extend(predicates);
            }
            ty::Predicate::WellFormed(..) => {
                // Currently, we do not elaborate WF predicates,
                // although we easily could.
            }
            ty::Predicate::ObjectSafe(..) => {
                // Currently, we do not elaborate object-safe
                // predicates.
            }
            ty::Predicate::Subtype(..) => {
                // Currently, we do not "elaborate" predicates like `X <: Y`,
                // though conceivably we might.
            }
            ty::Predicate::Projection(..) => {
                // Nothing to elaborate in a projection predicate.
            }
            ty::Predicate::ClosureKind(..) => {
                // Nothing to elaborate when waiting for a closure's kind to be inferred.
            }
            ty::Predicate::ConstEvaluatable(..) => {
                // Currently, we do not elaborate const-evaluatable
                // predicates.
            }
            ty::Predicate::RegionOutlives(..) => {
                // Nothing to elaborate from `'a: 'b`.
            }
            ty::Predicate::TypeOutlives(ref data) => {
                // We know that `T: 'a` for some type `T`. We can
                // often elaborate this. For example, if we know that
                // `[U]: 'a`, that implies that `U: 'a`. Similarly, if
                // we know `&'a U: 'b`, then we know that `'a: 'b` and
                // `U: 'b`.
                //
                // We can basically ignore bound regions here. So for
                // example `for<'c> Foo<'a,'c>: 'b` can be elaborated to
                // `'a: 'b`.

                // Ignore `for<'a> T: 'a` -- we might in the future
                // consider this as evidence that `T: 'static`, but
                // I'm a bit wary of such constructions and so for now
                // I want to be conservative. --nmatsakis
                let ty_max = data.skip_binder().0;
                let r_min = data.skip_binder().1;
                if r_min.is_late_bound() {
                    return;
                }

                let visited = &mut self.visited;
                let mut components = smallvec![];
                tcx.push_outlives_components(ty_max, &mut components);
                self.stack.extend(
                    components
                        .into_iter()
                        .filter_map(|component| match component {
                            Component::Region(r) => {
                                if r.is_late_bound() {
                                    None
                                } else {
                                    Some(ty::Predicate::RegionOutlives(ty::Binder::dummy(
                                        ty::OutlivesPredicate(r, r_min),
                                    )))
                                }
                            }

                            Component::Param(p) => {
                                let ty = tcx.mk_ty_param(p.index, p.name);
                                Some(ty::Predicate::TypeOutlives(ty::Binder::dummy(
                                    ty::OutlivesPredicate(ty, r_min),
                                )))
                            }

                            Component::UnresolvedInferenceVariable(_) => None,

                            Component::Projection(_) | Component::EscapingProjection(_) => {
                                // We can probably do more here. This
                                // corresponds to a case like `<T as
                                // Foo<'a>>::U: 'b`.
                                None
                            }
                        })
                        .filter(|p| visited.insert(p)),
                );
            }
        }
    }
}

impl Iterator for Elaborator<'tcx> {
    type Item = ty::Predicate<'tcx>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }

    fn next(&mut self) -> Option<ty::Predicate<'tcx>> {
        // Extract next item from top-most stack frame, if any.
        if let Some(pred) = self.stack.pop() {
            self.elaborate(&pred);
            Some(pred)
        } else {
            None
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Supertrait iterator
///////////////////////////////////////////////////////////////////////////

pub type Supertraits<'tcx> = FilterToTraits<Elaborator<'tcx>>;

pub fn supertraits<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> Supertraits<'tcx> {
    elaborate_trait_ref(tcx, trait_ref).filter_to_traits()
}

pub fn transitive_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    bounds: impl Iterator<Item = ty::PolyTraitRef<'tcx>>,
) -> Supertraits<'tcx> {
    elaborate_trait_refs(tcx, bounds).filter_to_traits()
}

///////////////////////////////////////////////////////////////////////////
// `TraitAliasExpander` iterator
///////////////////////////////////////////////////////////////////////////

/// "Trait alias expansion" is the process of expanding a sequence of trait
/// references into another sequence by transitively following all trait
/// aliases. e.g. If you have bounds like `Foo + Send`, a trait alias
/// `trait Foo = Bar + Sync;`, and another trait alias
/// `trait Bar = Read + Write`, then the bounds would expand to
/// `Read + Write + Sync + Send`.
/// Expansion is done via a DFS (depth-first search), and the `visited` field
/// is used to avoid cycles.
pub struct TraitAliasExpander<'tcx> {
    tcx: TyCtxt<'tcx>,
    stack: Vec<TraitAliasExpansionInfo<'tcx>>,
}

/// Stores information about the expansion of a trait via a path of zero or more trait aliases.
#[derive(Debug, Clone)]
pub struct TraitAliasExpansionInfo<'tcx> {
    pub path: SmallVec<[(ty::PolyTraitRef<'tcx>, Span); 4]>,
}

impl<'tcx> TraitAliasExpansionInfo<'tcx> {
    fn new(trait_ref: ty::PolyTraitRef<'tcx>, span: Span) -> Self {
        Self { path: smallvec![(trait_ref, span)] }
    }

    /// Adds diagnostic labels to `diag` for the expansion path of a trait through all intermediate
    /// trait aliases.
    pub fn label_with_exp_info(
        &self,
        diag: &mut DiagnosticBuilder<'_>,
        top_label: &str,
        use_desc: &str,
    ) {
        diag.span_label(self.top().1, top_label);
        if self.path.len() > 1 {
            for (_, sp) in self.path.iter().rev().skip(1).take(self.path.len() - 2) {
                diag.span_label(*sp, format!("referenced here ({})", use_desc));
            }
        }
        diag.span_label(
            self.bottom().1,
            format!("trait alias used in trait object type ({})", use_desc),
        );
    }

    pub fn trait_ref(&self) -> &ty::PolyTraitRef<'tcx> {
        &self.top().0
    }

    pub fn top(&self) -> &(ty::PolyTraitRef<'tcx>, Span) {
        self.path.last().unwrap()
    }

    pub fn bottom(&self) -> &(ty::PolyTraitRef<'tcx>, Span) {
        self.path.first().unwrap()
    }

    fn clone_and_push(&self, trait_ref: ty::PolyTraitRef<'tcx>, span: Span) -> Self {
        let mut path = self.path.clone();
        path.push((trait_ref, span));

        Self { path }
    }
}

pub fn expand_trait_aliases<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_refs: impl IntoIterator<Item = (ty::PolyTraitRef<'tcx>, Span)>,
) -> TraitAliasExpander<'tcx> {
    let items: Vec<_> = trait_refs
        .into_iter()
        .map(|(trait_ref, span)| TraitAliasExpansionInfo::new(trait_ref, span))
        .collect();
    TraitAliasExpander { tcx, stack: items }
}

impl<'tcx> TraitAliasExpander<'tcx> {
    /// If `item` is a trait alias and its predicate has not yet been visited, then expands `item`
    /// to the definition, pushes the resulting expansion onto `self.stack`, and returns `false`.
    /// Otherwise, immediately returns `true` if `item` is a regular trait, or `false` if it is a
    /// trait alias.
    /// The return value indicates whether `item` should be yielded to the user.
    fn expand(&mut self, item: &TraitAliasExpansionInfo<'tcx>) -> bool {
        let tcx = self.tcx;
        let trait_ref = item.trait_ref();
        let pred = trait_ref.without_const().to_predicate();

        debug!("expand_trait_aliases: trait_ref={:?}", trait_ref);

        // Don't recurse if this bound is not a trait alias.
        let is_alias = tcx.is_trait_alias(trait_ref.def_id());
        if !is_alias {
            return true;
        }

        // Don't recurse if this trait alias is already on the stack for the DFS search.
        let anon_pred = anonymize_predicate(tcx, &pred);
        if item.path.iter().rev().skip(1).any(|(tr, _)| {
            anonymize_predicate(tcx, &tr.without_const().to_predicate()) == anon_pred
        }) {
            return false;
        }

        // Get components of trait alias.
        let predicates = tcx.super_predicates_of(trait_ref.def_id());

        let items = predicates.predicates.iter().rev().filter_map(|(pred, span)| {
            pred.subst_supertrait(tcx, &trait_ref)
                .to_opt_poly_trait_ref()
                .map(|trait_ref| item.clone_and_push(trait_ref, *span))
        });
        debug!("expand_trait_aliases: items={:?}", items.clone());

        self.stack.extend(items);

        false
    }
}

impl<'tcx> Iterator for TraitAliasExpander<'tcx> {
    type Item = TraitAliasExpansionInfo<'tcx>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }

    fn next(&mut self) -> Option<TraitAliasExpansionInfo<'tcx>> {
        while let Some(item) = self.stack.pop() {
            if self.expand(&item) {
                return Some(item);
            }
        }
        None
    }
}

///////////////////////////////////////////////////////////////////////////
// Iterator over def-IDs of supertraits
///////////////////////////////////////////////////////////////////////////

pub struct SupertraitDefIds<'tcx> {
    tcx: TyCtxt<'tcx>,
    stack: Vec<DefId>,
    visited: FxHashSet<DefId>,
}

pub fn supertrait_def_ids(tcx: TyCtxt<'_>, trait_def_id: DefId) -> SupertraitDefIds<'_> {
    SupertraitDefIds {
        tcx,
        stack: vec![trait_def_id],
        visited: Some(trait_def_id).into_iter().collect(),
    }
}

impl Iterator for SupertraitDefIds<'tcx> {
    type Item = DefId;

    fn next(&mut self) -> Option<DefId> {
        let def_id = self.stack.pop()?;
        let predicates = self.tcx.super_predicates_of(def_id);
        let visited = &mut self.visited;
        self.stack.extend(
            predicates
                .predicates
                .iter()
                .filter_map(|(pred, _)| pred.to_opt_poly_trait_ref())
                .map(|trait_ref| trait_ref.def_id())
                .filter(|&super_def_id| visited.insert(super_def_id)),
        );
        Some(def_id)
    }
}

///////////////////////////////////////////////////////////////////////////
// Other
///////////////////////////////////////////////////////////////////////////

/// A filter around an iterator of predicates that makes it yield up
/// just trait references.
pub struct FilterToTraits<I> {
    base_iterator: I,
}

impl<I> FilterToTraits<I> {
    fn new(base: I) -> FilterToTraits<I> {
        FilterToTraits { base_iterator: base }
    }
}

impl<'tcx, I: Iterator<Item = ty::Predicate<'tcx>>> Iterator for FilterToTraits<I> {
    type Item = ty::PolyTraitRef<'tcx>;

    fn next(&mut self) -> Option<ty::PolyTraitRef<'tcx>> {
        while let Some(pred) = self.base_iterator.next() {
            if let ty::Predicate::Trait(data, _) = pred {
                return Some(data.to_poly_trait_ref());
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.base_iterator.size_hint();
        (0, upper)
    }
}

///////////////////////////////////////////////////////////////////////////
// Other
///////////////////////////////////////////////////////////////////////////

/// Instantiate all bound parameters of the impl with the given substs,
/// returning the resulting trait ref and all obligations that arise.
/// The obligations are closed under normalization.
pub fn impl_trait_ref_and_oblig<'a, 'tcx>(
    selcx: &mut SelectionContext<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl_def_id: DefId,
    impl_substs: SubstsRef<'tcx>,
) -> (ty::TraitRef<'tcx>, Vec<PredicateObligation<'tcx>>) {
    let impl_trait_ref = selcx.tcx().impl_trait_ref(impl_def_id).unwrap();
    let impl_trait_ref = impl_trait_ref.subst(selcx.tcx(), impl_substs);
    let Normalized { value: impl_trait_ref, obligations: normalization_obligations1 } =
        super::normalize(selcx, param_env, ObligationCause::dummy(), &impl_trait_ref);

    let predicates = selcx.tcx().predicates_of(impl_def_id);
    let predicates = predicates.instantiate(selcx.tcx(), impl_substs);
    let Normalized { value: predicates, obligations: normalization_obligations2 } =
        super::normalize(selcx, param_env, ObligationCause::dummy(), &predicates);
    let impl_obligations =
        predicates_for_generics(ObligationCause::dummy(), 0, param_env, &predicates);

    let impl_obligations: Vec<_> = impl_obligations
        .into_iter()
        .chain(normalization_obligations1)
        .chain(normalization_obligations2)
        .collect();

    (impl_trait_ref, impl_obligations)
}

/// See [`super::obligations_for_generics`].
pub fn predicates_for_generics<'tcx>(
    cause: ObligationCause<'tcx>,
    recursion_depth: usize,
    param_env: ty::ParamEnv<'tcx>,
    generic_bounds: &ty::InstantiatedPredicates<'tcx>,
) -> Vec<PredicateObligation<'tcx>> {
    debug!("predicates_for_generics(generic_bounds={:?})", generic_bounds);

    generic_bounds
        .predicates
        .iter()
        .map(|&predicate| Obligation {
            cause: cause.clone(),
            recursion_depth,
            param_env,
            predicate,
        })
        .collect()
}

pub fn predicate_for_trait_ref<'tcx>(
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    recursion_depth: usize,
) -> PredicateObligation<'tcx> {
    Obligation {
        cause,
        param_env,
        recursion_depth,
        predicate: trait_ref.without_const().to_predicate(),
    }
}

pub fn predicate_for_trait_def(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    trait_def_id: DefId,
    recursion_depth: usize,
    self_ty: Ty<'tcx>,
    params: &[GenericArg<'tcx>],
) -> PredicateObligation<'tcx> {
    let trait_ref =
        ty::TraitRef { def_id: trait_def_id, substs: tcx.mk_substs_trait(self_ty, params) };
    predicate_for_trait_ref(cause, param_env, trait_ref, recursion_depth)
}

/// Casts a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub fn upcast_choices(
    tcx: TyCtxt<'tcx>,
    source_trait_ref: ty::PolyTraitRef<'tcx>,
    target_trait_def_id: DefId,
) -> Vec<ty::PolyTraitRef<'tcx>> {
    if source_trait_ref.def_id() == target_trait_def_id {
        return vec![source_trait_ref]; // Shortcut the most common case.
    }

    supertraits(tcx, source_trait_ref).filter(|r| r.def_id() == target_trait_def_id).collect()
}

/// Given a trait `trait_ref`, returns the number of vtable entries
/// that come from `trait_ref`, excluding its supertraits. Used in
/// computing the vtable base for an upcast trait of a trait object.
pub fn count_own_vtable_entries(tcx: TyCtxt<'tcx>, trait_ref: ty::PolyTraitRef<'tcx>) -> usize {
    let mut entries = 0;
    // Count number of methods and add them to the total offset.
    // Skip over associated types and constants.
    for trait_item in tcx.associated_items(trait_ref.def_id()).in_definition_order() {
        if trait_item.kind == ty::AssocKind::Method {
            entries += 1;
        }
    }
    entries
}

/// Given an upcast trait object described by `object`, returns the
/// index of the method `method_def_id` (which should be part of
/// `object.upcast_trait_ref`) within the vtable for `object`.
pub fn get_vtable_index_of_object_method<N>(
    tcx: TyCtxt<'tcx>,
    object: &super::VtableObjectData<'tcx, N>,
    method_def_id: DefId,
) -> usize {
    // Count number of methods preceding the one we are selecting and
    // add them to the total offset.
    // Skip over associated types and constants.
    let mut entries = object.vtable_base;
    for trait_item in tcx.associated_items(object.upcast_trait_ref.def_id()).in_definition_order() {
        if trait_item.def_id == method_def_id {
            // The item with the ID we were given really ought to be a method.
            assert_eq!(trait_item.kind, ty::AssocKind::Method);
            return entries;
        }
        if trait_item.kind == ty::AssocKind::Method {
            entries += 1;
        }
    }

    bug!("get_vtable_index_of_object_method: {:?} was not found", method_def_id);
}

pub fn closure_trait_ref_and_return_type(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyFnSig<'tcx>,
    tuple_arguments: TupleArgumentsFlag,
) -> ty::Binder<(ty::TraitRef<'tcx>, Ty<'tcx>)> {
    let arguments_tuple = match tuple_arguments {
        TupleArgumentsFlag::No => sig.skip_binder().inputs()[0],
        TupleArgumentsFlag::Yes => tcx.intern_tup(sig.skip_binder().inputs()),
    };
    let trait_ref = ty::TraitRef {
        def_id: fn_trait_def_id,
        substs: tcx.mk_substs_trait(self_ty, &[arguments_tuple.into()]),
    };
    ty::Binder::bind((trait_ref, sig.skip_binder().output()))
}

pub fn generator_trait_ref_and_outputs(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyGenSig<'tcx>,
) -> ty::Binder<(ty::TraitRef<'tcx>, Ty<'tcx>, Ty<'tcx>)> {
    let trait_ref = ty::TraitRef {
        def_id: fn_trait_def_id,
        substs: tcx.mk_substs_trait(self_ty, &[sig.skip_binder().resume_ty.into()]),
    };
    ty::Binder::bind((trait_ref, sig.skip_binder().yield_ty, sig.skip_binder().return_ty))
}

pub fn impl_is_default(tcx: TyCtxt<'_>, node_item_def_id: DefId) -> bool {
    match tcx.hir().as_local_hir_id(node_item_def_id) {
        Some(hir_id) => {
            let item = tcx.hir().expect_item(hir_id);
            if let hir::ItemKind::Impl { defaultness, .. } = item.kind {
                defaultness.is_default()
            } else {
                false
            }
        }
        None => tcx.impl_defaultness(node_item_def_id).is_default(),
    }
}

pub fn impl_item_is_final(tcx: TyCtxt<'_>, assoc_item: &ty::AssocItem) -> bool {
    assoc_item.defaultness.is_final() && !impl_is_default(tcx, assoc_item.container.id())
}

pub enum TupleArgumentsFlag {
    Yes,
    No,
}
