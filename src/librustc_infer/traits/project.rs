//! Code for projecting associated types out of trait references.

use super::PredicateObligation;

use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, Ty};
use rustc_data_structures::snapshot_map::{Snapshot, SnapshotMap};

pub use rustc::traits::Reveal;

#[derive(Clone)]
pub struct MismatchedProjectionTypes<'tcx> {
    pub err: ty::error::TypeError<'tcx>,
}

#[derive(Clone, TypeFoldable)]
pub struct Normalized<'tcx, T> {
    pub value: T,
    pub obligations: Vec<PredicateObligation<'tcx>>,
}

pub type NormalizedTy<'tcx> = Normalized<'tcx, Ty<'tcx>>;

impl<'tcx, T> Normalized<'tcx, T> {
    pub fn with<U>(self, value: U) -> Normalized<'tcx, U> {
        Normalized { value: value, obligations: self.obligations }
    }
}

// # Cache

/// The projection cache. Unlike the standard caches, this can include
/// infcx-dependent type variables, therefore we have to roll the
/// cache back each time we roll a snapshot back, to avoid assumptions
/// on yet-unresolved inference variables. Types with placeholder
/// regions also have to be removed when the respective snapshot ends.
///
/// Because of that, projection cache entries can be "stranded" and left
/// inaccessible when type variables inside the key are resolved. We make no
/// attempt to recover or remove "stranded" entries, but rather let them be
/// (for the lifetime of the infcx).
///
/// Entries in the projection cache might contain inference variables
/// that will be resolved by obligations on the projection cache entry (e.g.,
/// when a type parameter in the associated type is constrained through
/// an "RFC 447" projection on the impl).
///
/// When working with a fulfillment context, the derived obligations of each
/// projection cache entry will be registered on the fulfillcx, so any users
/// that can wait for a fulfillcx fixed point need not care about this. However,
/// users that don't wait for a fixed point (e.g., trait evaluation) have to
/// resolve the obligations themselves to make sure the projected result is
/// ok and avoid issues like #43132.
///
/// If that is done, after evaluation the obligations, it is a good idea to
/// call `ProjectionCache::complete` to make sure the obligations won't be
/// re-evaluated and avoid an exponential worst-case.
//
// FIXME: we probably also want some sort of cross-infcx cache here to
// reduce the amount of duplication. Let's see what we get with the Chalk reforms.
#[derive(Default)]
pub struct ProjectionCache<'tcx> {
    map: SnapshotMap<ProjectionCacheKey<'tcx>, ProjectionCacheEntry<'tcx>>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProjectionCacheKey<'tcx> {
    ty: ty::ProjectionTy<'tcx>,
}

impl ProjectionCacheKey<'tcx> {
    pub fn new(ty: ty::ProjectionTy<'tcx>) -> Self {
        Self { ty }
    }
}

#[derive(Clone, Debug)]
pub enum ProjectionCacheEntry<'tcx> {
    InProgress,
    Ambiguous,
    Error,
    NormalizedTy(NormalizedTy<'tcx>),
}

// N.B., intentionally not Clone
pub struct ProjectionCacheSnapshot {
    snapshot: Snapshot,
}

impl<'tcx> ProjectionCache<'tcx> {
    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn snapshot(&mut self) -> ProjectionCacheSnapshot {
        ProjectionCacheSnapshot { snapshot: self.map.snapshot() }
    }

    pub fn rollback_to(&mut self, snapshot: ProjectionCacheSnapshot) {
        self.map.rollback_to(snapshot.snapshot);
    }

    pub fn rollback_placeholder(&mut self, snapshot: &ProjectionCacheSnapshot) {
        self.map.partial_rollback(&snapshot.snapshot, &|k| k.ty.has_re_placeholders());
    }

    pub fn commit(&mut self, snapshot: ProjectionCacheSnapshot) {
        self.map.commit(snapshot.snapshot);
    }

    /// Try to start normalize `key`; returns an error if
    /// normalization already occurred (this error corresponds to a
    /// cache hit, so it's actually a good thing).
    pub fn try_start(
        &mut self,
        key: ProjectionCacheKey<'tcx>,
    ) -> Result<(), ProjectionCacheEntry<'tcx>> {
        if let Some(entry) = self.map.get(&key) {
            return Err(entry.clone());
        }

        self.map.insert(key, ProjectionCacheEntry::InProgress);
        Ok(())
    }

    /// Indicates that `key` was normalized to `value`.
    pub fn insert_ty(&mut self, key: ProjectionCacheKey<'tcx>, value: NormalizedTy<'tcx>) {
        debug!(
            "ProjectionCacheEntry::insert_ty: adding cache entry: key={:?}, value={:?}",
            key, value
        );
        let fresh_key = self.map.insert(key, ProjectionCacheEntry::NormalizedTy(value));
        assert!(!fresh_key, "never started projecting `{:?}`", key);
    }

    /// Mark the relevant projection cache key as having its derived obligations
    /// complete, so they won't have to be re-computed (this is OK to do in a
    /// snapshot - if the snapshot is rolled back, the obligations will be
    /// marked as incomplete again).
    pub fn complete(&mut self, key: ProjectionCacheKey<'tcx>) {
        let ty = match self.map.get(&key) {
            Some(&ProjectionCacheEntry::NormalizedTy(ref ty)) => {
                debug!("ProjectionCacheEntry::complete({:?}) - completing {:?}", key, ty);
                ty.value
            }
            ref value => {
                // Type inference could "strand behind" old cache entries. Leave
                // them alone for now.
                debug!("ProjectionCacheEntry::complete({:?}) - ignoring {:?}", key, value);
                return;
            }
        };

        self.map.insert(
            key,
            ProjectionCacheEntry::NormalizedTy(Normalized { value: ty, obligations: vec![] }),
        );
    }

    /// A specialized version of `complete` for when the key's value is known
    /// to be a NormalizedTy.
    pub fn complete_normalized(&mut self, key: ProjectionCacheKey<'tcx>, ty: &NormalizedTy<'tcx>) {
        // We want to insert `ty` with no obligations. If the existing value
        // already has no obligations (as is common) we don't insert anything.
        if !ty.obligations.is_empty() {
            self.map.insert(
                key,
                ProjectionCacheEntry::NormalizedTy(Normalized {
                    value: ty.value,
                    obligations: vec![],
                }),
            );
        }
    }

    /// Indicates that trying to normalize `key` resulted in
    /// ambiguity. No point in trying it again then until we gain more
    /// type information (in which case, the "fully resolved" key will
    /// be different).
    pub fn ambiguous(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map.insert(key, ProjectionCacheEntry::Ambiguous);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }

    /// Indicates that trying to normalize `key` resulted in
    /// error.
    pub fn error(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map.insert(key, ProjectionCacheEntry::Error);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }
}
