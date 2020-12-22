use crate::ty::context::TyCtxt;
use crate::ty::{DefId, DefIdTree};
use rustc_hir::CRATE_HIR_ID;
use smallvec::SmallVec;
use std::mem;

/// Represents a forest of `DefId`s closed under the ancestor relation. That is,
/// if a `DefId` representing a module is contained in the forest then all
/// `DefId`s defined in that module or submodules are also implicitly contained
/// in the forest.
///
/// This is used to represent a set of modules in which a type is visibly
/// uninhabited.
#[derive(Clone)]
pub struct DefIdForest {
    /// The minimal set of `DefId`s required to represent the whole set.
    /// If A and B are DefIds in the `DefIdForest`, and A is a descendant
    /// of B, then only B will be in `root_ids`.
    /// We use a `SmallVec` here because (for its use for caching inhabitedness)
    /// it's rare that this will contain even two IDs.
    root_ids: SmallVec<[DefId; 1]>,
}

impl<'tcx> DefIdForest {
    /// Creates an empty forest.
    pub fn empty() -> DefIdForest {
        DefIdForest { root_ids: SmallVec::new() }
    }

    /// Creates a forest consisting of a single tree representing the entire
    /// crate.
    #[inline]
    pub fn full(tcx: TyCtxt<'tcx>) -> DefIdForest {
        let crate_id = tcx.hir().local_def_id(CRATE_HIR_ID);
        DefIdForest::from_id(crate_id.to_def_id())
    }

    /// Creates a forest containing a `DefId` and all its descendants.
    pub fn from_id(id: DefId) -> DefIdForest {
        let mut root_ids = SmallVec::new();
        root_ids.push(id);
        DefIdForest { root_ids }
    }

    /// Tests whether the forest is empty.
    pub fn is_empty(&self) -> bool {
        self.root_ids.is_empty()
    }

    /// Tests whether the forest contains a given DefId.
    pub fn contains(&self, tcx: TyCtxt<'tcx>, id: DefId) -> bool {
        self.root_ids.iter().any(|root_id| tcx.is_descendant_of(id, *root_id))
    }

    /// Calculate the intersection of a collection of forests.
    pub fn intersection<I>(tcx: TyCtxt<'tcx>, iter: I) -> DefIdForest
    where
        I: IntoIterator<Item = DefIdForest>,
    {
        let mut iter = iter.into_iter();
        let mut ret = if let Some(first) = iter.next() {
            first
        } else {
            return DefIdForest::full(tcx);
        };

        let mut next_ret = SmallVec::new();
        let mut old_ret: SmallVec<[DefId; 1]> = SmallVec::new();
        for next_forest in iter {
            // No need to continue if the intersection is already empty.
            if ret.is_empty() {
                break;
            }

            for id in ret.root_ids.drain(..) {
                if next_forest.contains(tcx, id) {
                    next_ret.push(id);
                } else {
                    old_ret.push(id);
                }
            }
            ret.root_ids.extend(old_ret.drain(..));

            next_ret.extend(next_forest.root_ids.into_iter().filter(|&id| ret.contains(tcx, id)));

            mem::swap(&mut next_ret, &mut ret.root_ids);
            next_ret.drain(..);
        }
        ret
    }

    /// Calculate the union of a collection of forests.
    pub fn union<I>(tcx: TyCtxt<'tcx>, iter: I) -> DefIdForest
    where
        I: IntoIterator<Item = DefIdForest>,
    {
        let mut ret = DefIdForest::empty();
        let mut next_ret = SmallVec::new();
        for next_forest in iter {
            next_ret.extend(ret.root_ids.drain(..).filter(|&id| !next_forest.contains(tcx, id)));

            for id in next_forest.root_ids {
                if !next_ret.contains(&id) {
                    next_ret.push(id);
                }
            }

            mem::swap(&mut next_ret, &mut ret.root_ids);
            next_ret.drain(..);
        }
        ret
    }
}
