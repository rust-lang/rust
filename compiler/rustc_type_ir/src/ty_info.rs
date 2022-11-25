use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    ops::Deref,
};

use rustc_data_structures::{
    fingerprint::Fingerprint,
    stable_hasher::{HashStable, StableHasher},
};

/// A helper type that you can wrap round your own type in order to automatically
/// cache the stable hash on creation and not recompute it whenever the stable hash
/// of the type is computed.
/// This is only done in incremental mode. You can also opt out of caching by using
/// StableHash::ZERO for the hash, in which case the hash gets computed each time.
/// This is useful if you have values that you intern but never (can?) use for stable
/// hashing.
#[derive(Copy, Clone)]
pub struct WithCachedTypeInfo<T> {
    pub internee: T,
    pub stable_hash: Fingerprint,
}

impl<T: PartialEq> PartialEq for WithCachedTypeInfo<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internee.eq(&other.internee)
    }
}

impl<T: Eq> Eq for WithCachedTypeInfo<T> {}

impl<T: Ord> PartialOrd for WithCachedTypeInfo<T> {
    fn partial_cmp(&self, other: &WithCachedTypeInfo<T>) -> Option<Ordering> {
        Some(self.internee.cmp(&other.internee))
    }
}

impl<T: Ord> Ord for WithCachedTypeInfo<T> {
    fn cmp(&self, other: &WithCachedTypeInfo<T>) -> Ordering {
        self.internee.cmp(&other.internee)
    }
}

impl<T> Deref for WithCachedTypeInfo<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.internee
    }
}

impl<T: Hash> Hash for WithCachedTypeInfo<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        if self.stable_hash != Fingerprint::ZERO {
            self.stable_hash.hash(s)
        } else {
            self.internee.hash(s)
        }
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for WithCachedTypeInfo<T> {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        if self.stable_hash == Fingerprint::ZERO || cfg!(debug_assertions) {
            // No cached hash available. This can only mean that incremental is disabled.
            // We don't cache stable hashes in non-incremental mode, because they are used
            // so rarely that the performance actually suffers.

            // We need to build the hash as if we cached it and then hash that hash, as
            // otherwise the hashes will differ between cached and non-cached mode.
            let stable_hash: Fingerprint = {
                let mut hasher = StableHasher::new();
                self.internee.hash_stable(hcx, &mut hasher);
                hasher.finish()
            };
            if cfg!(debug_assertions) && self.stable_hash != Fingerprint::ZERO {
                assert_eq!(
                    stable_hash, self.stable_hash,
                    "cached stable hash does not match freshly computed stable hash"
                );
            }
            stable_hash.hash_stable(hcx, hasher);
        } else {
            self.stable_hash.hash_stable(hcx, hasher);
        }
    }
}
