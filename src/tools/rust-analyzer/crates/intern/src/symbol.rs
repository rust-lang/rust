//! Attempt at flexible symbol interning, allowing to intern and free strings at runtime while also
//! supporting compile time declaration of symbols that will never be freed.

use std::{
    fmt,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    mem::{self, ManuallyDrop},
    ptr::NonNull,
    sync::OnceLock,
};

use dashmap::{DashMap, SharedValue};
use hashbrown::raw::RawTable;
use rustc_hash::FxHasher;
use triomphe::Arc;

pub mod symbols;

// some asserts for layout compatibility
const _: () = assert!(size_of::<Box<str>>() == size_of::<&str>());
const _: () = assert!(align_of::<Box<str>>() == align_of::<&str>());

const _: () = assert!(size_of::<Arc<Box<str>>>() == size_of::<&&str>());
const _: () = assert!(align_of::<Arc<Box<str>>>() == align_of::<&&str>());

const _: () = assert!(size_of::<*const *const str>() == size_of::<TaggedArcPtr>());
const _: () = assert!(align_of::<*const *const str>() == align_of::<TaggedArcPtr>());

const _: () = assert!(size_of::<Arc<Box<str>>>() == size_of::<TaggedArcPtr>());
const _: () = assert!(align_of::<Arc<Box<str>>>() == align_of::<TaggedArcPtr>());

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
        assert!(align_of::<&'static &'static str>().trailing_zeros() as usize > Self::BOOL_BITS);
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
        assert!(align_of::<&'static &'static str>().trailing_zeros() as usize > Self::BOOL_BITS);
        Self {
            packed: Self::pack_arc(
                // Safety: `Arc::into_raw` always returns a non null pointer
                unsafe { NonNull::new_unchecked(Arc::into_raw(arc).cast_mut().cast()) },
            ),
        }
    }

    /// Retrieves the tag.
    ///
    /// # Safety
    ///
    /// You can only drop the `Arc` if the instance is dropped.
    #[inline]
    pub(crate) unsafe fn try_as_arc_owned(self) -> Option<ManuallyDrop<Arc<Box<str>>>> {
        // Unpack the tag from the alignment niche
        let tag = self.packed.as_ptr().addr() & Self::BOOL_BITS;
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

        unsafe {
            // Safety: The pointer is derived from a non-null and bit-oring it with true (1) will
            // not make it null.
            NonNull::new_unchecked(ptr.as_ptr().map_addr(|addr| addr | packed_tag))
        }
    }

    #[inline]
    pub(crate) fn pointer(self) -> NonNull<*const str> {
        // SAFETY: The resulting pointer is guaranteed to be NonNull as we only modify the niche bytes
        unsafe {
            NonNull::new_unchecked(self.packed.as_ptr().map_addr(|addr| addr & !Self::BOOL_BITS))
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

const _: () = assert!(size_of::<Symbol>() == size_of::<NonNull<()>>());
const _: () = assert!(align_of::<Symbol>() == align_of::<NonNull<()>>());

type Map = DashMap<Symbol, (), BuildHasherDefault<FxHasher>>;
static MAP: OnceLock<Map> = OnceLock::new();

impl Symbol {
    pub fn intern(s: &str) -> Self {
        let storage = MAP.get_or_init(symbols::prefill);
        let (mut shard, hash) = Self::select_shard(storage, s);
        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, copy out its entry, conditionally bumping the backing Arc and return it
        //   - if not, put it into a box and then into an Arc, insert it, bump the ref-count and return the copy
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.
        let bucket = match shard.find_or_find_insert_slot(
            hash,
            |(other, _)| other.as_str() == s,
            |(x, _)| Self::hash(storage, x.as_str()),
        ) {
            Ok(bucket) => bucket,
            // SAFETY: The slot came from `find_or_find_insert_slot()`, and the table wasn't modified since then.
            Err(insert_slot) => unsafe {
                shard.insert_in_slot(
                    hash,
                    insert_slot,
                    (
                        Symbol { repr: TaggedArcPtr::arc(Arc::new(Box::<str>::from(s))) },
                        SharedValue::new(()),
                    ),
                )
            },
        };
        // SAFETY: We just retrieved/inserted this bucket.
        unsafe { bucket.as_ref().0.clone() }
    }

    pub fn integer(i: usize) -> Self {
        match i {
            0 => symbols::INTEGER_0,
            1 => symbols::INTEGER_1,
            2 => symbols::INTEGER_2,
            3 => symbols::INTEGER_3,
            4 => symbols::INTEGER_4,
            5 => symbols::INTEGER_5,
            6 => symbols::INTEGER_6,
            7 => symbols::INTEGER_7,
            8 => symbols::INTEGER_8,
            9 => symbols::INTEGER_9,
            10 => symbols::INTEGER_10,
            11 => symbols::INTEGER_11,
            12 => symbols::INTEGER_12,
            13 => symbols::INTEGER_13,
            14 => symbols::INTEGER_14,
            15 => symbols::INTEGER_15,
            i => Symbol::intern(&format!("{i}")),
        }
    }

    pub fn empty() -> Self {
        symbols::__empty
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.repr.as_str()
    }

    #[inline]
    fn select_shard(
        storage: &'static Map,
        s: &str,
    ) -> (dashmap::RwLockWriteGuard<'static, RawTable<(Symbol, SharedValue<()>)>>, u64) {
        let hash = Self::hash(storage, s);
        let shard_idx = storage.determine_shard(hash as usize);
        let shard = &storage.shards()[shard_idx];
        (shard.write(), hash)
    }

    #[inline]
    fn hash(storage: &'static Map, s: &str) -> u64 {
        storage.hasher().hash_one(s)
    }

    #[cold]
    fn drop_slow(arc: &Arc<Box<str>>) {
        let storage = MAP.get_or_init(symbols::prefill);
        let (mut shard, hash) = Self::select_shard(storage, arc);

        match Arc::count(arc) {
            0 | 1 => unreachable!(),
            2 => (),
            _ => {
                // Another thread has interned another copy
                return;
            }
        }

        let s = &***arc;
        let (ptr, _) = shard.remove_entry(hash, |(x, _)| x.as_str() == s).unwrap();
        let ptr = ManuallyDrop::new(ptr);
        // SAFETY: We're dropping, we have ownership.
        ManuallyDrop::into_inner(unsafe { ptr.repr.try_as_arc_owned().unwrap() });
        debug_assert_eq!(Arc::count(arc), 1);

        // Shrink the backing storage if the shard is less than 50% occupied.
        if shard.len() * 2 < shard.capacity() {
            let len = shard.len();
            shard.shrink_to(len, |(x, _)| Self::hash(storage, x.as_str()));
        }
    }
}

impl Drop for Symbol {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: We're dropping, we have ownership.
        let Some(arc) = (unsafe { self.repr.try_as_arc_owned() }) else {
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
    // SAFETY: We're not dropping the `Arc`.
    let Some(arc) = (unsafe { repr.try_as_arc_owned() }) else {
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
