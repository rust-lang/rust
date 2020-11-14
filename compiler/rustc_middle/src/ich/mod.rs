//! ICH - Incremental Compilation Hash

use crate::ty::{fast_reject, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::DefId;
use smallvec::SmallVec;
use std::cmp::Ord;

pub use rustc_crate::ich::{NodeIdHashingMode, StableHashingContext, StableHashingContextProvider};

mod impls_ty;

impl StableHashingContextProvider<'tcx> for TyCtxt<'tcx> {
    fn get_stable_hashing_context(&self) -> StableHashingContext<'tcx> {
        (*self).create_stable_hashing_context()
    }
}

pub fn hash_stable_trait_impls<'a>(
    hcx: &mut StableHashingContext<'a>,
    hasher: &mut StableHasher,
    blanket_impls: &[DefId],
    non_blanket_impls: &FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,
) {
    {
        let mut blanket_impls: SmallVec<[_; 8]> =
            blanket_impls.iter().map(|&def_id| hcx.def_path_hash(def_id)).collect();

        if blanket_impls.len() > 1 {
            blanket_impls.sort_unstable();
        }

        blanket_impls.hash_stable(hcx, hasher);
    }

    {
        let mut keys: SmallVec<[_; 8]> =
            non_blanket_impls.keys().map(|k| (k, k.map_def(|d| hcx.def_path_hash(d)))).collect();
        keys.sort_unstable_by(|&(_, ref k1), &(_, ref k2)| k1.cmp(k2));
        keys.len().hash_stable(hcx, hasher);
        for (key, ref stable_key) in keys {
            stable_key.hash_stable(hcx, hasher);
            let mut impls: SmallVec<[_; 8]> =
                non_blanket_impls[key].iter().map(|&impl_id| hcx.def_path_hash(impl_id)).collect();

            if impls.len() > 1 {
                impls.sort_unstable();
            }

            impls.hash_stable(hcx, hasher);
        }
    }
}
