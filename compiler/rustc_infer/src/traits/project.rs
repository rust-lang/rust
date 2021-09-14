//! Code for projecting associated types out of trait references.

use super::PredicateObligation;

use crate::infer::InferCtxtUndoLogs;

use rustc_data_structures::{
    snapshot_map::{self, SnapshotMapRef, SnapshotMapStorage},
    undo_log::Rollback,
};
use rustc_middle::ty::{self, Ty};

pub use rustc_middle::traits::Reveal;

pub(crate) type UndoLog<'tcx> =
    snapshot_map::UndoLog<ProjectionCacheKey<'tcx>, ProjectionCacheEntry<'tcx>>;

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
        Normalized { value, obligations: self.obligations }
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
pub struct ProjectionCache<'a, 'tcx> {
    map: &'a mut SnapshotMapStorage<ProjectionCacheKey<'tcx>, ProjectionCacheEntry<'tcx>>,
    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

#[derive(Default)]
pub struct ProjectionCacheStorage<'tcx> {
    map: SnapshotMapStorage<ProjectionCacheKey<'tcx>, ProjectionCacheEntry<'tcx>>,
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
    Recur,
    Error,
    NormalizedTy(NormalizedTy<'tcx>),
}

impl<'tcx> ProjectionCacheStorage<'tcx> {
    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> ProjectionCache<'a, 'tcx> {
        ProjectionCache { map: &mut self.map, undo_log }
    }
}

impl<'tcx> ProjectionCache<'_, 'tcx> {
    #[inline]
    fn map(
        &mut self,
    ) -> SnapshotMapRef<
        '_,
        ProjectionCacheKey<'tcx>,
        ProjectionCacheEntry<'tcx>,
        InferCtxtUndoLogs<'tcx>,
    > {
        self.map.with_log(self.undo_log)
    }

    pub fn clear(&mut self) {
        self.map().clear();
    }

    /// Try to start normalize `key`; returns an error if
    /// normalization already occurred (this error corresponds to a
    /// cache hit, so it's actually a good thing).
    pub fn try_start(
        &mut self,
        key: ProjectionCacheKey<'tcx>,
    ) -> Result<(), ProjectionCacheEntry<'tcx>> {
        let mut map = self.map();
        if let Some(entry) = map.get(&key) {
            return Err(entry.clone());
        }

        map.insert(key, ProjectionCacheEntry::InProgress);
        Ok(())
    }

    /// Indicates that `key` was normalized to `value`.
    pub fn insert_ty(&mut self, key: ProjectionCacheKey<'tcx>, value: NormalizedTy<'tcx>) {
        debug!(
            "ProjectionCacheEntry::insert_ty: adding cache entry: key={:?}, value={:?}",
            key, value
        );
        let mut map = self.map();
        if let Some(ProjectionCacheEntry::Recur) = map.get(&key) {
            debug!("Not overwriting Recur");
            return;
        }
        let fresh_key = map.insert(key, ProjectionCacheEntry::NormalizedTy(value));
        assert!(!fresh_key, "never started projecting `{:?}`", key);
    }

    /// Indicates that trying to normalize `key` resulted in
    /// ambiguity. No point in trying it again then until we gain more
    /// type information (in which case, the "fully resolved" key will
    /// be different).
    pub fn ambiguous(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map().insert(key, ProjectionCacheEntry::Ambiguous);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }

    /// Indicates that while trying to normalize `key`, `key` was required to
    /// be normalized again. Selection or evaluation should eventually report
    /// an error here.
    pub fn recur(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map().insert(key, ProjectionCacheEntry::Recur);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }

    /// Indicates that trying to normalize `key` resulted in
    /// error.
    pub fn error(&mut self, key: ProjectionCacheKey<'tcx>) {
        let fresh = self.map().insert(key, ProjectionCacheEntry::Error);
        assert!(!fresh, "never started projecting `{:?}`", key);
    }
}

impl<'tcx> Rollback<UndoLog<'tcx>> for ProjectionCacheStorage<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        self.map.reverse(undo);
    }
}
