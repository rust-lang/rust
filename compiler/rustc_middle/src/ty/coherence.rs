use std::fmt;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};

use crate::ty::{Predicate, SimplifiedType, TraitRef, Ty};

/// A map for the local crate mapping each type to a vector of its
/// inherent impls. This is not meant to be used outside of coherence;
/// rather, you should request the vector for a specific type via
/// `tcx.inherent_impls(def_id)` so as to minimize your dependencies
/// (constructing this map requires touching the entire crate).
#[derive(Clone, Debug, Default, HashStable)]
pub struct CrateInherentImpls {
    pub inherent_impls: LocalDefIdMap<Vec<DefId>>,
    pub incoherent_impls: FxHashMap<SimplifiedType, Vec<LocalDefId>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImplOverlapKind {
    /// These impls are always allowed to overlap.
    Permitted {
        /// Whether or not the impl is permitted due to the trait being a `#[marker]` trait
        marker: bool,
    },
    /// These impls are allowed to overlap, but that raises
    /// an issue #33140 future-compatibility warning.
    ///
    /// Some background: in Rust 1.0, the trait-object types `Send + Sync` (today's
    /// `dyn Send + Sync`) and `Sync + Send` (now `dyn Sync + Send`) were different.
    ///
    /// The widely-used version 0.1.0 of the crate `traitobject` had accidentally relied
    /// that difference, making what reduces to the following set of impls:
    ///
    /// ```compile_fail,(E0119)
    /// trait Trait {}
    /// impl Trait for dyn Send + Sync {}
    /// impl Trait for dyn Sync + Send {}
    /// ```
    ///
    /// Obviously, once we made these types be identical, that code causes a coherence
    /// error and a fairly big headache for us. However, luckily for us, the trait
    /// `Trait` used in this case is basically a marker trait, and therefore having
    /// overlapping impls for it is sound.
    ///
    /// To handle this, we basically regard the trait as a marker trait, with an additional
    /// future-compatibility warning. To avoid accidentally "stabilizing" this feature,
    /// it has the following restrictions:
    ///
    /// 1. The trait must indeed be a marker-like trait (i.e., no items), and must be
    /// positive impls.
    /// 2. The trait-ref of both impls must be equal.
    /// 3. The trait-ref of both impls must be a trait object type consisting only of
    /// marker traits.
    /// 4. Neither of the impls can have any where-clauses.
    ///
    /// Once `traitobject` 0.1.0 is no longer an active concern, this hack can be removed.
    Issue33140,
}

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds / where-clauses).
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TypeFoldable, TypeVisitable)]
pub enum ImplSubject<'tcx> {
    Trait(TraitRef<'tcx>),
    Inherent(Ty<'tcx>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable, Debug)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
    /// `#[rustc_reservation_impl] impl Trait for Type`
    ///
    /// This is a "stability hack", not a real Rust feature.
    /// See #64631 for details.
    Reservation,
}

impl ImplPolarity {
    /// Flips polarity by turning `Positive` into `Negative` and `Negative` into `Positive`.
    pub fn flip(&self) -> Option<ImplPolarity> {
        match self {
            ImplPolarity::Positive => Some(ImplPolarity::Negative),
            ImplPolarity::Negative => Some(ImplPolarity::Positive),
            ImplPolarity::Reservation => None,
        }
    }
}

impl fmt::Display for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => f.write_str("positive"),
            Self::Negative => f.write_str("negative"),
            Self::Reservation => f.write_str("reservation"),
        }
    }
}
