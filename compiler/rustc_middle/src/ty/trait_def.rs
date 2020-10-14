use crate::ich::{self, StableHashingContext};
use crate::traits::specialization_graph;
use crate::ty::fast_reject;
use crate::ty::fold::TypeFoldable;
use crate::ty::{Ty, TyCtxt};
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::HirId;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::ErrorReported;
use rustc_macros::HashStable;
use std::collections::BTreeMap;

/// A trait's definition with type information.
#[derive(HashStable)]
pub struct TraitDef {
    // We already have the def_path_hash below, no need to hash it twice
    #[stable_hasher(ignore)]
    pub def_id: DefId,

    pub unsafety: hir::Unsafety,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait will
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    pub has_auto_impl: bool,

    /// If `true`, then this trait has the `#[marker]` attribute, indicating
    /// that all its associated items have defaults that cannot be overridden,
    /// and thus `impl`s of it are allowed to overlap.
    pub is_marker: bool,

    /// Used to determine whether the standard library is allowed to specialize
    /// on this trait.
    pub specialization_kind: TraitSpecializationKind,

    /// The ICH of this trait's DefPath, cached here so it doesn't have to be
    /// recomputed all the time.
    pub def_path_hash: DefPathHash,
}

/// Whether this trait is treated specially by the standard library
/// specialization lint.
#[derive(HashStable, PartialEq, Clone, Copy, TyEncodable, TyDecodable)]
pub enum TraitSpecializationKind {
    /// The default. Specializing on this trait is not allowed.
    None,
    /// Specializing on this trait is allowed because it doesn't have any
    /// methods. For example `Sized` or `FusedIterator`.
    /// Applies to traits with the `rustc_unsafe_specialization_marker`
    /// attribute.
    Marker,
    /// Specializing on this trait is allowed because all of the impls of this
    /// trait are "always applicable". Always applicable means that if
    /// `X<'x>: T<'y>` for any lifetimes, then `for<'a, 'b> X<'a>: T<'b>`.
    /// Applies to traits with the `rustc_specialization_trait` attribute.
    AlwaysApplicable,
}

#[derive(Default)]
pub struct TraitImpls {
    blanket_impls: Vec<DefId>,
    /// Impls indexed by their simplified self type, for fast lookup.
    non_blanket_impls: FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,
}

impl<'tcx> TraitDef {
    pub fn new(
        def_id: DefId,
        unsafety: hir::Unsafety,
        paren_sugar: bool,
        has_auto_impl: bool,
        is_marker: bool,
        specialization_kind: TraitSpecializationKind,
        def_path_hash: DefPathHash,
    ) -> TraitDef {
        TraitDef {
            def_id,
            unsafety,
            paren_sugar,
            has_auto_impl,
            is_marker,
            specialization_kind,
            def_path_hash,
        }
    }

    pub fn ancestors(
        &self,
        tcx: TyCtxt<'tcx>,
        of_impl: DefId,
    ) -> Result<specialization_graph::Ancestors<'tcx>, ErrorReported> {
        specialization_graph::ancestors(tcx, self.def_id, of_impl)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn for_each_impl<F: FnMut(DefId)>(self, def_id: DefId, mut f: F) {
        let impls = self.trait_impls_of(def_id);

        for &impl_def_id in impls.blanket_impls.iter() {
            f(impl_def_id);
        }

        for v in impls.non_blanket_impls.values() {
            for &impl_def_id in v {
                f(impl_def_id);
            }
        }
    }

    /// Iterate over every impl that could possibly match the
    /// self type `self_ty`.
    pub fn for_each_relevant_impl<F: FnMut(DefId)>(
        self,
        def_id: DefId,
        self_ty: Ty<'tcx>,
        mut f: F,
    ) {
        let _: Option<()> = self.find_map_relevant_impl(def_id, self_ty, |did| {
            f(did);
            None
        });
    }

    /// Applies function to every impl that could possibly match the self type `self_ty` and returns
    /// the first non-none value.
    pub fn find_map_relevant_impl<T, F: FnMut(DefId) -> Option<T>>(
        self,
        def_id: DefId,
        self_ty: Ty<'tcx>,
        mut f: F,
    ) -> Option<T> {
        let impls = self.trait_impls_of(def_id);

        for &impl_def_id in impls.blanket_impls.iter() {
            if let result @ Some(_) = f(impl_def_id) {
                return result;
            }
        }

        // simplify_type(.., false) basically replaces type parameters and
        // projections with infer-variables. This is, of course, done on
        // the impl trait-ref when it is instantiated, but not on the
        // predicate trait-ref which is passed here.
        //
        // for example, if we match `S: Copy` against an impl like
        // `impl<T:Copy> Copy for Option<T>`, we replace the type variable
        // in `Option<T>` with an infer variable, to `Option<_>` (this
        // doesn't actually change fast_reject output), but we don't
        // replace `S` with anything - this impl of course can't be
        // selected, and as there are hundreds of similar impls,
        // considering them would significantly harm performance.

        // This depends on the set of all impls for the trait. That is
        // unfortunate. When we get red-green recompilation, we would like
        // to have a way of knowing whether the set of relevant impls
        // changed. The most naive
        // way would be to compute the Vec of relevant impls and see whether
        // it differs between compilations. That shouldn't be too slow by
        // itself - we do quite a bit of work for each relevant impl anyway.
        //
        // If we want to be faster, we could have separate queries for
        // blanket and non-blanket impls, and compare them separately.
        //
        // I think we'll cross that bridge when we get to it.
        if let Some(simp) = fast_reject::simplify_type(self, self_ty, true) {
            if let Some(impls) = impls.non_blanket_impls.get(&simp) {
                for &impl_def_id in impls {
                    if let result @ Some(_) = f(impl_def_id) {
                        return result;
                    }
                }
            }
        } else {
            for &impl_def_id in impls.non_blanket_impls.values().flatten() {
                if let result @ Some(_) = f(impl_def_id) {
                    return result;
                }
            }
        }

        None
    }

    /// Returns an iterator containing all impls
    pub fn all_impls(self, def_id: DefId) -> impl Iterator<Item = DefId> + 'tcx {
        let TraitImpls { blanket_impls, non_blanket_impls } = self.trait_impls_of(def_id);

        blanket_impls.iter().chain(non_blanket_impls.iter().map(|(_, v)| v).flatten()).cloned()
    }
}

// Query provider for `all_local_trait_impls`.
pub(super) fn all_local_trait_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    krate: CrateNum,
) -> &'tcx BTreeMap<DefId, Vec<HirId>> {
    &tcx.hir_crate(krate).trait_impls
}

// Query provider for `trait_impls_of`.
pub(super) fn trait_impls_of_provider(tcx: TyCtxt<'_>, trait_id: DefId) -> TraitImpls {
    let mut impls = TraitImpls::default();

    // Traits defined in the current crate can't have impls in upstream
    // crates, so we don't bother querying the cstore.
    if !trait_id.is_local() {
        for &cnum in tcx.crates().iter() {
            for &(impl_def_id, simplified_self_ty) in
                tcx.implementations_of_trait((cnum, trait_id)).iter()
            {
                if let Some(simplified_self_ty) = simplified_self_ty {
                    impls
                        .non_blanket_impls
                        .entry(simplified_self_ty)
                        .or_default()
                        .push(impl_def_id);
                } else {
                    impls.blanket_impls.push(impl_def_id);
                }
            }
        }
    }

    for &hir_id in tcx.hir().trait_impls(trait_id) {
        let impl_def_id = tcx.hir().local_def_id(hir_id).to_def_id();

        let impl_self_ty = tcx.type_of(impl_def_id);
        if impl_self_ty.references_error() {
            continue;
        }

        if let Some(simplified_self_ty) = fast_reject::simplify_type(tcx, impl_self_ty, false) {
            impls.non_blanket_impls.entry(simplified_self_ty).or_default().push(impl_def_id);
        } else {
            impls.blanket_impls.push(impl_def_id);
        }
    }

    impls
}

impl<'a> HashStable<StableHashingContext<'a>> for TraitImpls {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let TraitImpls { ref blanket_impls, ref non_blanket_impls } = *self;

        ich::hash_stable_trait_impls(hcx, hasher, blanket_impls, non_blanket_impls);
    }
}
