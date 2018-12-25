use std::hash::Hash;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::collections::HashMap;
use std::collections::hash_map::RawEntryMut;
use std::borrow::Borrow;

pub trait HashInterner<K: Eq + Hash> {
    fn intern_ref<Q: ?Sized, F: FnOnce() -> K>(&mut self, value: &Q, make: F) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq;

    fn intern<Q, F: FnOnce(Q) -> K>(&mut self, value: Q, make: F) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq;
}

impl<K: Eq + Hash + Copy, S: BuildHasher> HashInterner<K> for HashMap<K, (), S> {
    #[inline]
    fn intern_ref<Q: ?Sized, F: FnOnce() -> K>(&mut self, value: &Q, make: F) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        let mut hasher = self.hasher().build_hasher();
        value.hash(&mut hasher);
        let hash = hasher.finish();
        let entry = self.raw_entry_mut().from_key_hashed_nocheck(hash, value);

        match entry {
            RawEntryMut::Occupied(e) => *e.key(),
            RawEntryMut::Vacant(e) => {
                let v = make();
                e.insert_hashed_nocheck(hash, v, ());
                v
            }
        }
    }

    #[inline]
    fn intern<Q, F: FnOnce(Q) -> K>(&mut self, value: Q, make: F) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        let mut hasher = self.hasher().build_hasher();
        value.hash(&mut hasher);
        let hash = hasher.finish();
        let entry = self.raw_entry_mut().from_key_hashed_nocheck(hash, &value);

        match entry {
            RawEntryMut::Occupied(e) => *e.key(),
            RawEntryMut::Vacant(e) => {
                let v = make(value);
                e.insert_hashed_nocheck(hash, v, ());
                v
            }
        }
    }
}
