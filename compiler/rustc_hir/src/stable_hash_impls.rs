use rustc_data_structures::stable_hasher::{StableHash, StableHashCtxt, StableHasher};

use crate::HashIgnoredAttrId;
use crate::hir::{AttributeMap, OwnerNodes};

// The following implementations of StableHash for `ItemId`, `TraitItemId`, and
// `ImplItemId` deserve special attention. Normally we do not hash `NodeId`s within
// the HIR, since they just signify a HIR nodes own path. But `ItemId` et al
// are used when another item in the HIR is *referenced* and we certainly
// want to pick up on a reference changing its target, so we hash the NodeIds
// in "DefPath Mode".

impl<'tcx> StableHash for OwnerNodes<'tcx> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        // We ignore the `nodes` and `bodies` fields since these refer to information included in
        // `hash` which is hashed in the collector and used for the crate hash.
        // `local_id_to_def_id` is also ignored because is dependent on the body, then just hashing
        // the body satisfies the condition of two nodes being different have different
        // `stable_hash` results.
        let OwnerNodes { opt_hash_including_bodies, nodes: _, bodies: _ } = *self;
        opt_hash_including_bodies.unwrap().stable_hash(hcx, hasher);
    }
}

impl<'tcx> StableHash for AttributeMap<'tcx> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        // We ignore the `map` since it refers to information included in `opt_hash` which is
        // hashed in the collector and used for the crate hash.
        let AttributeMap { opt_hash, define_opaque: _, map: _ } = *self;
        opt_hash.unwrap().stable_hash(hcx, hasher);
    }
}

impl StableHash for HashIgnoredAttrId {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {
        /* we don't hash HashIgnoredAttrId, we ignore them */
    }
}
