use crate::ty::context::TyCtxt;
use crate::ty::{DefId, DefIdTree};
use rustc_hir::CRATE_HIR_ID;
use smallvec::SmallVec;
use std::mem;
use std::sync::Arc;

use DefIdForest::*;

/// Represents a forest of `DefId`s closed under the ancestor relation. That is,
/// if a `DefId` representing a module is contained in the forest then all
/// `DefId`s defined in that module or submodules are also implicitly contained
/// in the forest.
///
/// This is used to represent a set of modules in which a type is visibly
/// uninhabited.
///
/// We store the minimal set of `DefId`s required to represent the whole set. If A and B are
/// `DefId`s in the `DefIdForest`, and A is a parent of B, then only A will be stored. When this is
/// used with `type_uninhabited_from`, there will very rarely be more than one `DefId` stored.
#[derive(Clone, HashStable, Debug)]
pub enum DefIdForest {
    Empty,
    Single(DefId),
    /// This variant is very rare.
    /// Invariant: >1 elements
    /// We use `Arc` because this is used in the output of a query.
    Multiple(Arc<[DefId]>),
}

/// Tests whether a slice of roots contains a given DefId.
#[inline]
fn slice_contains<'tcx>(tcx: TyCtxt<'tcx>, slice: &[DefId], id: DefId) -> bool {
    slice.iter().any(|root_id| tcx.is_descendant_of(id, *root_id))
}

impl<'tcx> DefIdForest {
    /// Creates an empty forest.
    pub fn empty() -> DefIdForest {
        DefIdForest::Empty
    }

    /// Creates a forest consisting of a single tree representing the entire
    /// crate.
    #[inline]
    pub fn full(tcx: TyCtxt<'tcx>) -> DefIdForest {
        DefIdForest::from_id(tcx.hir().local_def_id(CRATE_HIR_ID).to_def_id())
    }

    /// Creates a forest containing a `DefId` and all its descendants.
    pub fn from_id(id: DefId) -> DefIdForest {
        DefIdForest::Single(id)
    }

    fn as_slice(&self) -> &[DefId] {
        match self {
            Empty => &[],
            Single(id) => std::slice::from_ref(id),
            Multiple(root_ids) => root_ids,
        }
    }

    // Only allocates in the rare `Multiple` case.
    fn from_slice(root_ids: &[DefId]) -> DefIdForest {
        match root_ids {
            [] => Empty,
            [id] => Single(*id),
            _ => DefIdForest::Multiple(root_ids.into()),
        }
    }

    /// Tests whether the forest is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Empty => true,
            Single(..) | Multiple(..) => false,
        }
    }

    /// Iterate over the set of roots.
    fn iter(&self) -> impl Iterator<Item = DefId> + '_ {
        self.as_slice().iter().copied()
    }

    /// Tests whether the forest contains a given DefId.
    pub fn contains(&self, tcx: TyCtxt<'tcx>, id: DefId) -> bool {
        slice_contains(tcx, self.as_slice(), id)
    }

    /// Calculate the intersection of a collection of forests.
    pub fn intersection<I>(tcx: TyCtxt<'tcx>, iter: I) -> DefIdForest
    where
        I: IntoIterator<Item = DefIdForest>,
    {
        let mut iter = iter.into_iter();
        let mut ret: SmallVec<[_; 1]> = if let Some(first) = iter.next() {
            SmallVec::from_slice(first.as_slice())
        } else {
            return DefIdForest::full(tcx);
        };

        let mut next_ret: SmallVec<[_; 1]> = SmallVec::new();
        for next_forest in iter {
            // No need to continue if the intersection is already empty.
            if ret.is_empty() || next_forest.is_empty() {
                return DefIdForest::empty();
            }

            // We keep the elements in `ret` that are also in `next_forest`.
            next_ret.extend(ret.iter().copied().filter(|&id| next_forest.contains(tcx, id)));
            // We keep the elements in `next_forest` that are also in `ret`.
            next_ret.extend(next_forest.iter().filter(|&id| slice_contains(tcx, &ret, id)));

            mem::swap(&mut next_ret, &mut ret);
            next_ret.clear();
        }
        DefIdForest::from_slice(&ret)
    }

    /// Calculate the union of a collection of forests.
    pub fn union<I>(tcx: TyCtxt<'tcx>, iter: I) -> DefIdForest
    where
        I: IntoIterator<Item = DefIdForest>,
    {
        let mut ret: SmallVec<[_; 1]> = SmallVec::new();
        let mut next_ret: SmallVec<[_; 1]> = SmallVec::new();
        for next_forest in iter {
            // Union with the empty set is a no-op.
            if next_forest.is_empty() {
                continue;
            }

            // We add everything in `ret` that is not in `next_forest`.
            next_ret.extend(ret.iter().copied().filter(|&id| !next_forest.contains(tcx, id)));
            // We add everything in `next_forest` that we haven't added yet.
            for id in next_forest.iter() {
                if !slice_contains(tcx, &next_ret, id) {
                    next_ret.push(id);
                }
            }

            mem::swap(&mut next_ret, &mut ret);
            next_ret.clear();
        }
        DefIdForest::from_slice(&ret)
    }
}
