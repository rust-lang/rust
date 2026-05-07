//! This module contains `StableHash` implementations for various data types
//! from `rustc_middle::ty` in no particular order.

use std::cell::RefCell;
use std::ptr;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{
    HashingControls, StableHash, StableHashCtxt, StableHasher,
};
use tracing::trace;

use crate::{mir, ty};

impl<'tcx, H, T> StableHash for &'tcx ty::list::RawList<H, T>
where
    T: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        // Note: this cache makes an *enormous* performance difference on certain benchmarks. E.g.
        // without it, compiling `diesel-2.2.10` can be 74% slower, and compiling
        // `deeply-nested-multi` can be ~4,000x slower(!)
        thread_local! {
            static CACHE: RefCell<FxHashMap<(*const (), HashingControls), Fingerprint>> =
                RefCell::new(Default::default());
        }

        let hash = CACHE.with(|cache| {
            let key = (ptr::from_ref(*self).cast::<()>(), hcx.hashing_controls());
            if let Some(&hash) = cache.borrow().get(&key) {
                return hash;
            }

            let mut hasher = StableHasher::new();
            self[..].stable_hash(hcx, &mut hasher);

            let hash: Fingerprint = hasher.finish();
            cache.borrow_mut().insert(key, hash);
            hash
        });

        hash.stable_hash(hcx, hasher);
    }
}

impl<'tcx> StableHash for ty::GenericArg<'tcx> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.kind().stable_hash(hcx, hasher);
    }
}

// AllocIds get resolved to whatever they point to (to be stable)
impl StableHash for mir::interpret::AllocId {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.try_get_global_alloc(*self).stable_hash(hcx, hasher);
        });
    }
}

impl StableHash for mir::interpret::CtfeProvenance {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.into_parts().stable_hash(hcx, hasher);
    }
}
