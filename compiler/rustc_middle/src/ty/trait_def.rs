use crate::traits::specialization_graph;
use crate::ty::fast_reject::{self, SimplifiedType, TreatParams, TreatProjections};
use crate::ty::visit::TypeVisitableExt;
use crate::ty::{Ident, Ty, TyCtxt};
use hir::def_id::LOCAL_CRATE;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use std::iter;

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::ErrorGuaranteed;
use rustc_macros::HashStable;

/// A trait's definition with type information.
#[derive(HashStable, Encodable, Decodable)]
pub struct TraitDef {
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

    /// If `true`, then this trait has to `#[rustc_coinductive]` attribute or
    /// is an auto trait. This indicates that trait solver cycles involving an
    /// `X: ThisTrait` goal are accepted.
    ///
    /// In the future all traits should be coinductive, but we need a better
    /// formal understanding of what exactly that means and should probably
    /// also have already switched to the new trait solver.
    pub is_coinductive: bool,

    /// If `true`, then this trait has the `#[rustc_skip_array_during_method_dispatch]`
    /// attribute, indicating that editions before 2021 should not consider this trait
    /// during method dispatch if the receiver is an array.
    pub skip_array_during_method_dispatch: bool,

    /// Used to determine whether the standard library is allowed to specialize
    /// on this trait.
    pub specialization_kind: TraitSpecializationKind,

    /// List of functions from `#[rustc_must_implement_one_of]` attribute one of which
    /// must be implemented.
    pub must_implement_one_of: Option<Box<[Ident]>>,

    /// Whether to add a builtin `dyn Trait: Trait` implementation.
    /// This is enabled for all traits except ones marked with
    /// `#[rustc_deny_explicit_impl(implement_via_object = false)]`.
    pub implement_via_object: bool,

    /// Whether a trait is fully built-in, and any implementation is disallowed.
    /// This only applies to built-in traits, and is marked via
    /// `#[rustc_deny_explicit_impl(implement_via_object = ...)]`.
    pub deny_explicit_impl: bool,
}

/// Whether this trait is treated specially by the standard library
/// specialization lint.
#[derive(HashStable, PartialEq, Clone, Copy, Encodable, Decodable)]
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

#[derive(Default, Debug, HashStable)]
pub struct TraitImpls {
    blanket_impls: Vec<DefId>,
    /// Impls indexed by their simplified self type, for fast lookup.
    non_blanket_impls: FxIndexMap<SimplifiedType, Vec<DefId>>,
}

impl TraitImpls {
    pub fn blanket_impls(&self) -> &[DefId] {
        self.blanket_impls.as_slice()
    }

    pub fn non_blanket_impls(&self) -> &FxIndexMap<SimplifiedType, Vec<DefId>> {
        &self.non_blanket_impls
    }
}

impl<'tcx> TraitDef {
    pub fn ancestors(
        &self,
        tcx: TyCtxt<'tcx>,
        of_impl: DefId,
    ) -> Result<specialization_graph::Ancestors<'tcx>, ErrorGuaranteed> {
        specialization_graph::ancestors(tcx, self.def_id, of_impl)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn for_each_impl<F: FnMut(DefId)>(self, trait_def_id: DefId, mut f: F) {
        let impls = self.trait_impls_of(trait_def_id);

        for &impl_def_id in impls.blanket_impls.iter() {
            f(impl_def_id);
        }

        for v in impls.non_blanket_impls.values() {
            for &impl_def_id in v {
                f(impl_def_id);
            }
        }
    }

    /// Iterate over every impl that could possibly match the self type `self_ty`.
    ///
    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn for_each_relevant_impl(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        f: impl FnMut(DefId),
    ) {
        self.for_each_relevant_impl_treating_projections(
            trait_def_id,
            self_ty,
            TreatProjections::ForLookup,
            f,
        )
    }

    pub fn for_each_relevant_impl_treating_projections(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        treat_projections: TreatProjections,
        mut f: impl FnMut(DefId),
    ) {
        // FIXME: This depends on the set of all impls for the trait. That is
        // unfortunate wrt. incremental compilation.
        //
        // If we want to be faster, we could have separate queries for
        // blanket and non-blanket impls, and compare them separately.
        let impls = self.trait_impls_of(trait_def_id);

        for &impl_def_id in impls.blanket_impls.iter() {
            f(impl_def_id);
        }

        // Note that we're using `TreatParams::ForLookup` to query `non_blanket_impls` while using
        // `TreatParams::AsCandidateKey` while actually adding them.
        let treat_params = match treat_projections {
            TreatProjections::NextSolverLookup => TreatParams::NextSolverLookup,
            TreatProjections::ForLookup => TreatParams::ForLookup,
        };
        // This way, when searching for some impl for `T: Trait`, we do not look at any impls
        // whose outer level is not a parameter or projection. Especially for things like
        // `T: Clone` this is incredibly useful as we would otherwise look at all the impls
        // of `Clone` for `Option<T>`, `Vec<T>`, `ConcreteType` and so on.
        if let Some(simp) = fast_reject::simplify_type(self, self_ty, treat_params) {
            if let Some(impls) = impls.non_blanket_impls.get(&simp) {
                for &impl_def_id in impls {
                    f(impl_def_id);
                }
            }
        } else {
            for &impl_def_id in impls.non_blanket_impls.values().flatten() {
                f(impl_def_id);
            }
        }
    }

    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn non_blanket_impls_for_ty(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
    ) -> impl Iterator<Item = DefId> + 'tcx {
        let impls = self.trait_impls_of(trait_def_id);
        if let Some(simp) = fast_reject::simplify_type(self, self_ty, TreatParams::AsCandidateKey) {
            if let Some(impls) = impls.non_blanket_impls.get(&simp) {
                return impls.iter().copied();
            }
        }

        [].iter().copied()
    }

    /// Returns an iterator containing all impls for `trait_def_id`.
    ///
    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn all_impls(self, trait_def_id: DefId) -> impl Iterator<Item = DefId> + 'tcx {
        let TraitImpls { blanket_impls, non_blanket_impls } = self.trait_impls_of(trait_def_id);

        blanket_impls.iter().chain(non_blanket_impls.iter().flat_map(|(_, v)| v)).cloned()
    }
}

/// Query provider for `trait_impls_of`.
pub(super) fn trait_impls_of_provider(tcx: TyCtxt<'_>, trait_id: DefId) -> TraitImpls {
    let mut impls = TraitImpls::default();

    // Traits defined in the current crate can't have impls in upstream
    // crates, so we don't bother querying the cstore.
    if !trait_id.is_local() {
        for &cnum in tcx.crates(()).iter() {
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

    for &impl_def_id in tcx.hir().trait_impls(trait_id) {
        let impl_def_id = impl_def_id.to_def_id();

        let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();
        if impl_self_ty.references_error() {
            continue;
        }

        if let Some(simplified_self_ty) =
            fast_reject::simplify_type(tcx, impl_self_ty, TreatParams::AsCandidateKey)
        {
            impls.non_blanket_impls.entry(simplified_self_ty).or_default().push(impl_def_id);
        } else {
            impls.blanket_impls.push(impl_def_id);
        }
    }

    impls
}

/// Query provider for `incoherent_impls`.
pub(super) fn incoherent_impls_provider(tcx: TyCtxt<'_>, simp: SimplifiedType) -> &[DefId] {
    let mut impls = Vec::new();

    for cnum in iter::once(LOCAL_CRATE).chain(tcx.crates(()).iter().copied()) {
        for &impl_def_id in tcx.crate_incoherent_impls((cnum, simp)) {
            impls.push(impl_def_id)
        }
    }

    debug!(?impls);

    tcx.arena.alloc_slice(&impls)
}
