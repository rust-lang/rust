//! This module contains `HashStable` implementations for various data types
//! from `rustc_middle::ty` in no particular order.

use std::cell::RefCell;
use std::ptr;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{
    HashStable, HashStableContext, HashingControls, StableHasher, ToStableHashKey,
};
use tracing::trace;

use crate::middle::region;
use crate::{mir, ty};

impl<'tcx, H, T> HashStable for &'tcx ty::list::RawList<H, T>
where
    T: HashStable,
{
    fn hash_stable<Hcx: HashStableContext>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
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
            self[..].hash_stable(hcx, &mut hasher);

            let hash: Fingerprint = hasher.finish();
            cache.borrow_mut().insert(key, hash);
            hash
        });

        hash.hash_stable(hcx, hasher);
    }
}

impl<'tcx, H, T> ToStableHashKey for &'tcx ty::list::RawList<H, T>
where
    T: HashStable,
{
    type KeyType = Fingerprint;

    #[inline]
    fn to_stable_hash_key<Hcx: HashStableContext>(&self, hcx: &mut Hcx) -> Fingerprint {
        let mut hasher = StableHasher::new();
        self.hash_stable(hcx, &mut hasher);
        hasher.finish()
    }
}

impl<'tcx> HashStable for ty::GenericArg<'tcx> {
    fn hash_stable<Hcx: HashStableContext>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.kind().hash_stable(hcx, hasher);
    }
}

// AllocIds get resolved to whatever they point to (to be stable)
impl HashStable for mir::interpret::AllocId {
    fn hash_stable<Hcx: HashStableContext>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.try_get_global_alloc(*self).hash_stable(hcx, hasher);
        });
    }
}

impl HashStable for mir::interpret::CtfeProvenance {
    fn hash_stable<Hcx: HashStableContext>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.into_parts().hash_stable(hcx, hasher);
    }
}

impl ToStableHashKey for region::Scope {
    type KeyType = region::Scope;

    #[inline]
    fn to_stable_hash_key<Hcx>(&self, _: &mut Hcx) -> region::Scope {
        *self
    }
}
