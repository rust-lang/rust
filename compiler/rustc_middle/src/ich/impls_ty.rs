//! This module contains `HashStable` implementations for various data types
//! from `rustc_middle::ty` in no particular order.

use crate::ich::{NodeIdHashingMode, StableHashingContext};
use crate::middle::region;
use crate::mir;
use crate::ty;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use std::cell::RefCell;
use std::mem;

impl<'a, 'tcx, T> HashStable<StableHashingContext<'a>> for &'tcx ty::List<T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<(usize, usize), Fingerprint>> =
                RefCell::new(Default::default());
        }

        let hash = CACHE.with(|cache| {
            let key = (self.as_ptr() as usize, self.len());
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

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ty::subst::GenericArg<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.unpack().hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::RegionKind {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::ReErased | ty::ReStatic => {
                // No variant fields to hash for these ...
            }
            ty::ReEmpty(universe) => {
                universe.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrAnon(i)) => {
                db.hash_stable(hcx, hasher);
                i.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrNamed(def_id, name)) => {
                db.hash_stable(hcx, hasher);
                def_id.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrEnv) => {
                db.hash_stable(hcx, hasher);
            }
            ty::ReEarlyBound(ty::EarlyBoundRegion { def_id, index, name }) => {
                def_id.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
            }
            ty::ReFree(ref free_region) => {
                free_region.hash_stable(hcx, hasher);
            }
            ty::ReVar(..) | ty::RePlaceholder(..) => {
                bug!("StableHasher: unexpected region {:?}", *self)
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::RegionVid {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ty::ConstVid<'tcx> {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.index.hash_stable(hcx, hasher);
    }
}

impl<'tcx> HashStable<StableHashingContext<'tcx>> for ty::BoundVar {
    #[inline]
    fn hash_stable(&self, hcx: &mut StableHashingContext<'tcx>, hasher: &mut StableHasher) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a, T> HashStable<StableHashingContext<'a>> for ty::Binder<T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.as_ref().skip_binder().hash_stable(hcx, hasher);
    }
}

// AllocIds get resolved to whatever they point to (to be stable)
impl<'a> HashStable<StableHashingContext<'a>> for mir::interpret::AllocId {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.get_global_alloc(*self).hash_stable(hcx, hasher);
        });
    }
}

// `Relocations` with default type parameters is a sorted map.
impl<'a, Tag> HashStable<StableHashingContext<'a>> for mir::interpret::Relocations<Tag>
where
    Tag: HashStable<StableHashingContext<'a>>,
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

impl<'a> HashStable<StableHashingContext<'a>> for ty::TyVid {
    fn hash_stable(&self, _hcx: &mut StableHashingContext<'a>, _hasher: &mut StableHasher) {
        // `TyVid` values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TyKind::hash_stable() - can't hash a TyVid {:?}.", *self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::IntVid {
    fn hash_stable(&self, _hcx: &mut StableHashingContext<'a>, _hasher: &mut StableHasher) {
        // `IntVid` values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TyKind::hash_stable() - can't hash an IntVid {:?}.", *self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::FloatVid {
    fn hash_stable(&self, _hcx: &mut StableHashingContext<'a>, _hasher: &mut StableHasher) {
        // `FloatVid` values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TyKind::hash_stable() - can't hash a FloatVid {:?}.", *self)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for crate::middle::privacy::AccessLevels {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let crate::middle::privacy::AccessLevels { ref map } = *self;

            map.hash_stable(hcx, hasher);
        });
    }
}
