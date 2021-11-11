use rustc_errors::DiagnosticBuilder;
use rustc_span::Span;
use smallvec::smallvec;
use smallvec::SmallVec;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::subst::{GenericArg, Subst, SubstsRef};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt, TypeFoldable};

use super::{Normalized, Obligation, ObligationCause, PredicateObligation, SelectionContext};
pub use rustc_infer::traits::{self, util::*};

use std::iter;

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
        if self.top().1 != self.bottom().1 {
            // When the trait object is in a return type these two spans match, we don't want
            // redundant labels.
            diag.span_label(
                self.bottom().1,
                format!("trait alias used in trait object type ({})", use_desc),
            );
        }
    }

    pub fn trait_ref(&self) -> ty::PolyTraitRef<'tcx> {
        self.top().0
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
    trait_refs: impl Iterator<Item = (ty::PolyTraitRef<'tcx>, Span)>,
) -> TraitAliasExpander<'tcx> {
    let items: Vec<_> =
        trait_refs.map(|(trait_ref, span)| TraitAliasExpansionInfo::new(trait_ref, span)).collect();
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
        let pred = trait_ref.without_const().to_predicate(tcx);

        debug!("expand_trait_aliases: trait_ref={:?}", trait_ref);

        // Don't recurse if this bound is not a trait alias.
        let is_alias = tcx.is_trait_alias(trait_ref.def_id());
        if !is_alias {
            return true;
        }

        // Don't recurse if this trait alias is already on the stack for the DFS search.
        let anon_pred = anonymize_predicate(tcx, pred);
        if item.path.iter().rev().skip(1).any(|&(tr, _)| {
            anonymize_predicate(tcx, tr.without_const().to_predicate(tcx)) == anon_pred
        }) {
            return false;
        }

        // Get components of trait alias.
        let predicates = tcx.super_predicates_of(trait_ref.def_id());

        let items = predicates.predicates.iter().rev().filter_map(|(pred, span)| {
            pred.subst_supertrait(tcx, &trait_ref)
                .to_opt_poly_trait_pred()
                .map(|trait_ref| item.clone_and_push(trait_ref.map_bound(|t| t.trait_ref), *span))
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

impl Iterator for SupertraitDefIds<'_> {
    type Item = DefId;

    fn next(&mut self) -> Option<DefId> {
        let def_id = self.stack.pop()?;
        let predicates = self.tcx.super_predicates_of(def_id);
        let visited = &mut self.visited;
        self.stack.extend(
            predicates
                .predicates
                .iter()
                .filter_map(|(pred, _)| pred.to_opt_poly_trait_pred())
                .map(|trait_ref| trait_ref.def_id())
                .filter(|&super_def_id| visited.insert(super_def_id)),
        );
        Some(def_id)
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
) -> (ty::TraitRef<'tcx>, impl Iterator<Item = PredicateObligation<'tcx>>) {
    let impl_trait_ref = selcx.tcx().impl_trait_ref(impl_def_id).unwrap();
    let impl_trait_ref = impl_trait_ref.subst(selcx.tcx(), impl_substs);
    let Normalized { value: impl_trait_ref, obligations: normalization_obligations1 } =
        super::normalize(selcx, param_env, ObligationCause::dummy(), impl_trait_ref);

    let predicates = selcx.tcx().predicates_of(impl_def_id);
    let predicates = predicates.instantiate(selcx.tcx(), impl_substs);
    let Normalized { value: predicates, obligations: normalization_obligations2 } =
        super::normalize(selcx, param_env, ObligationCause::dummy(), predicates);
    let impl_obligations =
        predicates_for_generics(ObligationCause::dummy(), 0, param_env, predicates);

    let impl_obligations = impl_obligations
        .chain(normalization_obligations1.into_iter())
        .chain(normalization_obligations2.into_iter());

    (impl_trait_ref, impl_obligations)
}

pub fn predicates_for_generics<'tcx>(
    cause: ObligationCause<'tcx>,
    recursion_depth: usize,
    param_env: ty::ParamEnv<'tcx>,
    generic_bounds: ty::InstantiatedPredicates<'tcx>,
) -> impl Iterator<Item = PredicateObligation<'tcx>> {
    debug!("predicates_for_generics(generic_bounds={:?})", generic_bounds);

    iter::zip(generic_bounds.predicates, generic_bounds.spans).map(move |(predicate, span)| {
        let cause = match *cause.code() {
            traits::ItemObligation(def_id) if !span.is_dummy() => traits::ObligationCause::new(
                cause.span,
                cause.body_id,
                traits::BindingObligation(def_id, span),
            ),
            _ => cause.clone(),
        };
        Obligation { cause, recursion_depth, param_env, predicate }
    })
}

pub fn predicate_for_trait_ref<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    recursion_depth: usize,
) -> PredicateObligation<'tcx> {
    Obligation {
        cause,
        param_env,
        recursion_depth,
        predicate: ty::Binder::dummy(trait_ref).without_const().to_predicate(tcx),
    }
}

pub fn predicate_for_trait_def<'tcx>(
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
    predicate_for_trait_ref(tcx, cause, param_env, trait_ref, recursion_depth)
}

/// Casts a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub fn upcast_choices<'tcx>(
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
pub fn count_own_vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> usize {
    let existential_trait_ref =
        trait_ref.map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));
    let existential_trait_ref = tcx.erase_regions(existential_trait_ref);
    tcx.own_existential_vtable_entries(existential_trait_ref).len()
}

/// Given an upcast trait object described by `object`, returns the
/// index of the method `method_def_id` (which should be part of
/// `object.upcast_trait_ref`) within the vtable for `object`.
pub fn get_vtable_index_of_object_method<'tcx, N>(
    tcx: TyCtxt<'tcx>,
    object: &super::ImplSourceObjectData<'tcx, N>,
    method_def_id: DefId,
) -> usize {
    let existential_trait_ref = object
        .upcast_trait_ref
        .map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));
    let existential_trait_ref = tcx.erase_regions(existential_trait_ref);
    // Count number of methods preceding the one we are selecting and
    // add them to the total offset.
    let index = tcx
        .own_existential_vtable_entries(existential_trait_ref)
        .iter()
        .copied()
        .position(|def_id| def_id == method_def_id)
        .unwrap_or_else(|| {
            bug!("get_vtable_index_of_object_method: {:?} was not found", method_def_id);
        });
    object.vtable_base + index
}

pub fn closure_trait_ref_and_return_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyFnSig<'tcx>,
    tuple_arguments: TupleArgumentsFlag,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>)> {
    let arguments_tuple = match tuple_arguments {
        TupleArgumentsFlag::No => sig.skip_binder().inputs()[0],
        TupleArgumentsFlag::Yes => tcx.intern_tup(sig.skip_binder().inputs()),
    };
    debug_assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef {
        def_id: fn_trait_def_id,
        substs: tcx.mk_substs_trait(self_ty, &[arguments_tuple.into()]),
    };
    sig.map_bound(|sig| (trait_ref, sig.output()))
}

pub fn generator_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyGenSig<'tcx>,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>, Ty<'tcx>)> {
    debug_assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef {
        def_id: fn_trait_def_id,
        substs: tcx.mk_substs_trait(self_ty, &[sig.skip_binder().resume_ty.into()]),
    };
    sig.map_bound(|sig| (trait_ref, sig.yield_ty, sig.return_ty))
}

pub fn impl_item_is_final(tcx: TyCtxt<'_>, assoc_item: &ty::AssocItem) -> bool {
    assoc_item.defaultness.is_final() && tcx.impl_defaultness(assoc_item.container.id()).is_final()
}

pub enum TupleArgumentsFlag {
    Yes,
    No,
}
