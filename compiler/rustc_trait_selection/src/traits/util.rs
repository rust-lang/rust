use super::NormalizeExt;
use super::{ObligationCause, PredicateObligation, SelectionContext};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferOk;
use rustc_middle::ty::SubstsRef;
use rustc_middle::ty::{self, ImplSubject, ToPredicate, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::Span;
use smallvec::SmallVec;

pub use rustc_infer::traits::{self, util::*};

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
    pub fn label_with_exp_info(&self, diag: &mut Diagnostic, top_label: &str, use_desc: &str) {
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
        let predicates = tcx.implied_predicates_of(trait_ref.def_id());
        debug!(?predicates);

        let items = predicates.predicates.iter().rev().filter_map(|(pred, span)| {
            pred.subst_supertrait(tcx, &trait_ref)
                .to_opt_poly_trait_pred()
                .map(|trait_ref| item.clone_and_push(trait_ref.map_bound(|t| t.trait_ref), *span))
        });
        debug!("expand_trait_aliases: items={:?}", items.clone().collect::<Vec<_>>());

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

/// Instantiate all bound parameters of the impl subject with the given substs,
/// returning the resulting subject and all obligations that arise.
/// The obligations are closed under normalization.
pub fn impl_subject_and_oblig<'a, 'tcx>(
    selcx: &mut SelectionContext<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    impl_def_id: DefId,
    impl_substs: SubstsRef<'tcx>,
    cause: impl Fn(usize, Span) -> ObligationCause<'tcx>,
) -> (ImplSubject<'tcx>, impl Iterator<Item = PredicateObligation<'tcx>>) {
    let subject = selcx.tcx().impl_subject(impl_def_id);
    let subject = subject.subst(selcx.tcx(), impl_substs);

    let InferOk { value: subject, obligations: normalization_obligations1 } =
        selcx.infcx.at(&ObligationCause::dummy(), param_env).normalize(subject);

    let predicates = selcx.tcx().predicates_of(impl_def_id);
    let predicates = predicates.instantiate(selcx.tcx(), impl_substs);
    let InferOk { value: predicates, obligations: normalization_obligations2 } =
        selcx.infcx.at(&ObligationCause::dummy(), param_env).normalize(predicates);
    let impl_obligations = super::predicates_for_generics(cause, param_env, predicates);

    let impl_obligations = impl_obligations
        .chain(normalization_obligations1.into_iter())
        .chain(normalization_obligations2.into_iter());

    (subject, impl_obligations)
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

/// Given an upcast trait object described by `object`, returns the
/// index of the method `method_def_id` (which should be part of
/// `object.upcast_trait_ref`) within the vtable for `object`.
pub fn get_vtable_index_of_object_method<'tcx, N>(
    tcx: TyCtxt<'tcx>,
    object: &super::ImplSourceObjectData<'tcx, N>,
    method_def_id: DefId,
) -> Option<usize> {
    // Count number of methods preceding the one we are selecting and
    // add them to the total offset.
    tcx.own_existential_vtable_entries(object.upcast_trait_ref.def_id())
        .iter()
        .copied()
        .position(|def_id| def_id == method_def_id)
        .map(|index| object.vtable_base + index)
}

pub fn closure_trait_ref_and_return_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyFnSig<'tcx>,
    tuple_arguments: TupleArgumentsFlag,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>)> {
    assert!(!self_ty.has_escaping_bound_vars());
    let arguments_tuple = match tuple_arguments {
        TupleArgumentsFlag::No => sig.skip_binder().inputs()[0],
        TupleArgumentsFlag::Yes => tcx.mk_tup(sig.skip_binder().inputs()),
    };
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty, arguments_tuple]);
    sig.map_bound(|sig| (trait_ref, sig.output()))
}

pub fn generator_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyGenSig<'tcx>,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>, Ty<'tcx>)> {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty, sig.skip_binder().resume_ty]);
    sig.map_bound(|sig| (trait_ref, sig.yield_ty, sig.return_ty))
}

pub fn future_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyGenSig<'tcx>,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>)> {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty]);
    sig.map_bound(|sig| (trait_ref, sig.return_ty))
}

pub fn impl_item_is_final(tcx: TyCtxt<'_>, assoc_item: &ty::AssocItem) -> bool {
    assoc_item.defaultness(tcx).is_final()
        && tcx.impl_defaultness(assoc_item.container_id(tcx)).is_final()
}

pub enum TupleArgumentsFlag {
    Yes,
    No,
}
