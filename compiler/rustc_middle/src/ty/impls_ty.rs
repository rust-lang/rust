//! This module contains `HashStable` implementations for various data types
//! from `rustc_middle::ty` in no particular order.

use crate::middle::region;
use crate::mir;
use crate::ty;
use crate::ty::fast_reject::SimplifiedType;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::HashingControls;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_query_system::ich::StableHashingContext;
use std::cell::RefCell;

impl<'a, 'tcx, T> HashStable<StableHashingContext<'a>> for &'tcx ty::List<T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<(usize, usize, HashingControls), Fingerprint>> =
                RefCell::new(Default::default());
        }

        let hash = CACHE.with(|cache| {
            let key = (self.as_ptr() as usize, self.len(), hcx.hashing_controls());
            if let Some(&hash) = cache.borrow().get(&key) {
                return hash;
            }

            let mut hasher = StableHasher::new();
            (&self[..]).hash_stable(hcx, &mut hasher);

            let hash: Fingerprint = hasher.finish();
            cache.borrow_mut().insert(key, hash);
            hash
        });

        hash.hash_stable(hcx, hasher);
    }
}

impl<'a, 'tcx, T> ToStableHashKey<StableHashingContext<'a>> for &'tcx ty::List<T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    type KeyType = Fingerprint;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> Fingerprint {
        let mut hasher = StableHasher::new();
        let mut hcx: StableHashingContext<'a> = hcx.clone();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish()
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for SimplifiedType {
    type KeyType = Fingerprint;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> Fingerprint {
        let mut hasher = StableHasher::new();
        let mut hcx: StableHashingContext<'a> = hcx.clone();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish()
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ty::subst::GenericArg<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.unpack().hash_stable(hcx, hasher);
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ty::subst::GenericArgKind<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        match self {
            // WARNING: We dedup cache the `HashStable` results for `List`
            // while ignoring types and freely transmute
            // between `List<Ty<'tcx>>` and `List<GenericArg<'tcx>>`.
            // See `fn intern_type_list` for more details.
            //
            // We therefore hash types without adding a hash for their discriminant.
            //
            // In order to make it very unlikely for the sequence of bytes being hashed for
            // a `GenericArgKind::Type` to be the same as the sequence of bytes being
            // hashed for one of the other variants, we hash some very high number instead
            // of their actual discriminant since `TyKind` should never start with anything
            // that high.
            ty::subst::GenericArgKind::Type(ty) => ty.hash_stable(hcx, hasher),
            ty::subst::GenericArgKind::Const(ct) => {
                0xF3u8.hash_stable(hcx, hasher);
                ct.hash_stable(hcx, hasher);
            }
            ty::subst::GenericArgKind::Lifetime(lt) => {
                0xF5u8.hash_stable(hcx, hasher);
                lt.hash_stable(hcx, hasher);
            }
        }
    }
}

// AllocIds get resolved to whatever they point to (to be stable)
impl<'a> HashStable<StableHashingContext<'a>> for mir::interpret::AllocId {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.try_get_global_alloc(*self).hash_stable(hcx, hasher);
        });
    }
}

// `Relocations` with default type parameters is a sorted map.
impl<'a, Prov> HashStable<StableHashingContext<'a>> for mir::interpret::ProvenanceMap<Prov>
where
    Prov: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for reloc in self.iter() {
            reloc.hash_stable(hcx, hasher);
        }
    }
}

impl<'a> ToStableHashKey<StableHashingContext<'a>> for region::Scope {
    type KeyType = region::Scope;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'a>) -> region::Scope {
        *self
    }
}
