//! Attempt at flexible symbol interning, allowing to intern and free strings at runtime while also
//! supporting compile time declaration of symbols that will never be freed.

use std::{
    borrow::Borrow,
    fmt,
    hash::{BuildHasherDefault, Hash, Hasher},
    mem::{self, ManuallyDrop},
    ptr::NonNull,
    sync::OnceLock,
};

use dashmap::{DashMap, SharedValue};
use hashbrown::{hash_map::RawEntryMut, HashMap};
use rustc_hash::FxHasher;
use sptr::Strict;
use triomphe::Arc;

pub mod symbols;

// some asserts for layout compatibility
const _: () = assert!(std::mem::size_of::<Box<str>>() == std::mem::size_of::<&str>());
const _: () = assert!(std::mem::align_of::<Box<str>>() == std::mem::align_of::<&str>());

const _: () = assert!(std::mem::size_of::<Arc<Box<str>>>() == std::mem::size_of::<&&str>());
const _: () = assert!(std::mem::align_of::<Arc<Box<str>>>() == std::mem::align_of::<&&str>());

const _: () =
    assert!(std::mem::size_of::<*const *const str>() == std::mem::size_of::<TaggedArcPtr>());
const _: () =
    assert!(std::mem::align_of::<*const *const str>() == std::mem::align_of::<TaggedArcPtr>());

const _: () = assert!(std::mem::size_of::<Arc<Box<str>>>() == std::mem::size_of::<TaggedArcPtr>());
const _: () =
    assert!(std::mem::align_of::<Arc<Box<str>>>() == std::mem::align_of::<TaggedArcPtr>());

/// A pointer that points to a pointer to a `str`, it may be backed as a `&'static &'static str` or
/// `Arc<Box<str>>` but its size is that of a thin pointer. The active variant is encoded as a tag
/// in the LSB of the alignment niche.
// Note, Ideally this would encode a `ThinArc<str>` and `ThinRef<str>`/`ThinConstPtr<str>` instead of the double indirection.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct TaggedArcPtr {
    packed: NonNull<*const str>,
}

unsafe impl Send for TaggedArcPtr {}
unsafe impl Sync for TaggedArcPtr {}

impl TaggedArcPtr {
    const BOOL_BITS: usize = true as usize;

    const fn non_arc(r: &'static &'static str) -> Self {
        assert!(
            mem::align_of::<&'static &'static str>().trailing_zeros() as usize > Self::BOOL_BITS
        );
        // SAFETY: The pointer is non-null as it is derived from a reference
        // Ideally we would call out to `pack_arc` but for a `false` tag, unfortunately the
        // packing stuff requires reading out the pointer to an integer which is not supported
        // in const contexts, so here we make use of the fact that for the non-arc version the
        // tag is false (0) and thus does not need touching the actual pointer value.ext)

        let packed =
            unsafe { NonNull::new_unchecked((r as *const &str).cast::<*const str>().cast_mut()) };
        Self { packed }
    }

    fn arc(arc: Arc<Box<str>>) -> Self {
        assert!(
            mem::align_of::<&'static &'static str>().trailing_zeros() as usize > Self::BOOL_BITS
        );
        Self {
            packed: Self::pack_arc(
                // Safety: `Arc::into_raw` always returns a non null pointer
                unsafe { NonNull::new_unchecked(Arc::into_raw(arc).cast_mut().cast()) },
            ),
        }
    }

    /// Retrieves the tag.
    #[inline]
    pub(crate) fn try_as_arc_owned(self) -> Option<ManuallyDrop<Arc<Box<str>>>> {
        // Unpack the tag from the alignment niche
        let tag = Strict::addr(self.packed.as_ptr()) & Self::BOOL_BITS;
        if tag != 0 {
            // Safety: We checked that the tag is non-zero -> true, so we are pointing to the data offset of an `Arc`
            Some(ManuallyDrop::new(unsafe {
                Arc::from_raw(self.pointer().as_ptr().cast::<Box<str>>())
            }))
        } else {
            None
        }
    }

    #[inline]
    fn pack_arc(ptr: NonNull<*const str>) -> NonNull<*const str> {
        let packed_tag = true as usize;

        // can't use this strict provenance stuff here due to trait methods not being const
        // unsafe {
        //     // Safety: The pointer is derived from a non-null
        //     NonNull::new_unchecked(Strict::map_addr(ptr.as_ptr(), |addr| {
        //         // Safety:
        //         // - The pointer is `NonNull` => it's address is `NonZero<usize>`
        //         // - `P::BITS` least significant bits are always zero (`Pointer` contract)
        //         // - `T::BITS <= P::BITS` (from `Self::ASSERTION`)
        //         //
        //         // Thus `addr >> T::BITS` is guaranteed to be non-zero.
        //         //
        //         // `{non_zero} | packed_tag` can't make the value zero.

        //         (addr >> Self::BOOL_BITS) | packed_tag
        //     }))
        // }
        // so what follows is roughly what the above looks like but inlined

        let self_addr = ptr.as_ptr() as *const *const str as usize;
        let addr = self_addr | packed_tag;
        let dest_addr = addr as isize;
        let offset = dest_addr.wrapping_sub(self_addr as isize);

        // SAFETY: The resulting pointer is guaranteed to be NonNull as we only modify the niche bytes
        unsafe { NonNull::new_unchecked(ptr.as_ptr().cast::<u8>().wrapping_offset(offset).cast()) }
    }

    #[inline]
    pub(crate) fn pointer(self) -> NonNull<*const str> {
        // SAFETY: The resulting pointer is guaranteed to be NonNull as we only modify the niche bytes
        unsafe {
            NonNull::new_unchecked(Strict::map_addr(self.packed.as_ptr(), |addr| {
                addr & !Self::BOOL_BITS
            }))
        }
    }

    #[inline]
    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: We always point to a pointer to a str no matter what variant is active
        unsafe { *self.pointer().as_ptr().cast::<&str>() }
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct Symbol {
    repr: TaggedArcPtr,
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

const _: () = assert!(std::mem::size_of::<Symbol>() == std::mem::size_of::<NonNull<()>>());
const _: () = assert!(std::mem::align_of::<Symbol>() == std::mem::align_of::<NonNull<()>>());

static MAP: OnceLock<DashMap<SymbolProxy, (), BuildHasherDefault<FxHasher>>> = OnceLock::new();

impl Symbol {
    pub fn intern(s: &str) -> Self {
        let (mut shard, hash) = Self::select_shard(s);
        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, copy out its entry, conditionally bumping the backing Arc and return it
        //   - if not, put it into a box and then into an Arc, insert it, bump the ref-count and return the copy
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.
        match shard.raw_entry_mut().from_key_hashed_nocheck(hash, s) {
            RawEntryMut::Occupied(occ) => Self { repr: increase_arc_refcount(occ.key().0) },
            RawEntryMut::Vacant(vac) => Self {
                repr: increase_arc_refcount(
                    vac.insert_hashed_nocheck(
                        hash,
                        SymbolProxy(TaggedArcPtr::arc(Arc::new(Box::<str>::from(s)))),
                        SharedValue::new(()),
                    )
                    .0
                     .0,
                ),
            },
        }
    }

    pub fn integer(i: usize) -> Self {
        match i {
            0 => symbols::INTEGER_0.clone(),
            1 => symbols::INTEGER_1.clone(),
            2 => symbols::INTEGER_2.clone(),
            3 => symbols::INTEGER_3.clone(),
            4 => symbols::INTEGER_4.clone(),
            5 => symbols::INTEGER_5.clone(),
            6 => symbols::INTEGER_6.clone(),
            7 => symbols::INTEGER_7.clone(),
            8 => symbols::INTEGER_8.clone(),
            9 => symbols::INTEGER_9.clone(),
            10 => symbols::INTEGER_10.clone(),
            11 => symbols::INTEGER_11.clone(),
            12 => symbols::INTEGER_12.clone(),
            13 => symbols::INTEGER_13.clone(),
            14 => symbols::INTEGER_14.clone(),
            15 => symbols::INTEGER_15.clone(),
            i => Symbol::intern(&format!("{i}")),
        }
    }

    pub fn empty() -> Self {
        symbols::__empty.clone()
    }

    pub fn as_str(&self) -> &str {
        self.repr.as_str()
    }

    #[inline]
    fn select_shard(
        s: &str,
    ) -> (
        dashmap::RwLockWriteGuard<
            'static,
            HashMap<SymbolProxy, SharedValue<()>, BuildHasherDefault<FxHasher>>,
        >,
        u64,
    ) {
        let storage = MAP.get_or_init(symbols::prefill);
        let hash = {
            let mut hasher = std::hash::BuildHasher::build_hasher(storage.hasher());
            s.hash(&mut hasher);
            hasher.finish()
        };
        let shard_idx = storage.determine_shard(hash as usize);
        let shard = &storage.shards()[shard_idx];
        (shard.write(), hash)
    }

    #[cold]
    fn drop_slow(arc: &Arc<Box<str>>) {
        let (mut shard, hash) = Self::select_shard(arc);

        match Arc::count(arc) {
            0 => unreachable!(),
            1 => unreachable!(),
            2 => (),
            _ => {
                // Another thread has interned another copy
                return;
            }
        }

        ManuallyDrop::into_inner(
            match shard.raw_entry_mut().from_key_hashed_nocheck::<str>(hash, arc.as_ref()) {
                RawEntryMut::Occupied(occ) => occ.remove_entry(),
                RawEntryMut::Vacant(_) => unreachable!(),
            }
            .0
             .0
            .try_as_arc_owned()
            .unwrap(),
        );
        debug_assert_eq!(Arc::count(arc), 1);

        // Shrink the backing storage if the shard is less than 50% occupied.
        if shard.len() * 2 < shard.capacity() {
            shard.shrink_to_fit();
        }
    }
}

impl Drop for Symbol {
    #[inline]
    fn drop(&mut self) {
        let Some(arc) = self.repr.try_as_arc_owned() else {
            return;
        };
        // When the last `Ref` is dropped, remove the object from the global map.
        if Arc::count(&arc) == 2 {
            // Only `self` and the global map point to the object.

            Self::drop_slow(&arc);
        }
        // decrement the ref count
        ManuallyDrop::into_inner(arc);
    }
}

impl Clone for Symbol {
    fn clone(&self) -> Self {
        Self { repr: increase_arc_refcount(self.repr) }
    }
}

fn increase_arc_refcount(repr: TaggedArcPtr) -> TaggedArcPtr {
    let Some(arc) = repr.try_as_arc_owned() else {
        return repr;
    };
    // increase the ref count
    mem::forget(Arc::clone(&arc));
    repr
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

// only exists so we can use `from_key_hashed_nocheck` with a &str
#[derive(Debug, PartialEq, Eq)]
struct SymbolProxy(TaggedArcPtr);

impl Hash for SymbolProxy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_str().hash(state);
    }
}

impl Borrow<str> for SymbolProxy {
    fn borrow(&self) -> &str {
        self.0.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test() {
        Symbol::intern("isize");
        let base_len = MAP.get().unwrap().len();
        let hello = Symbol::intern("hello");
        let world = Symbol::intern("world");
        let more_worlds = world.clone();
        let bang = Symbol::intern("!");
        let q = Symbol::intern("?");
        assert_eq!(MAP.get().unwrap().len(), base_len + 4);
        let bang2 = Symbol::intern("!");
        assert_eq!(MAP.get().unwrap().len(), base_len + 4);
        drop(bang2);
        assert_eq!(MAP.get().unwrap().len(), base_len + 4);
        drop(q);
        assert_eq!(MAP.get().unwrap().len(), base_len + 3);
        let default = Symbol::intern("default");
        let many_worlds = world.clone();
        assert_eq!(MAP.get().unwrap().len(), base_len + 3);
        assert_eq!(
            "hello default world!",
            format!("{} {} {}{}", hello.as_str(), default.as_str(), world.as_str(), bang.as_str())
        );
        drop(default);
        assert_eq!(
            "hello world!",
            format!("{} {}{}", hello.as_str(), world.as_str(), bang.as_str())
        );
        drop(many_worlds);
        drop(more_worlds);
        drop(hello);
        drop(world);
        drop(bang);
        assert_eq!(MAP.get().unwrap().len(), base_len);
    }
}
