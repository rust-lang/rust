//! Interning of slices, potentially with a header.

use std::{
    borrow::Borrow,
    ffi::c_void,
    fmt::{self, Debug},
    hash::{BuildHasher, BuildHasherDefault, Hash, Hasher},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::Deref,
    ptr::{self, NonNull},
    sync::OnceLock,
};

use dashmap::{DashMap, SharedValue};
use hashbrown::raw::RawTable;
use rustc_hash::FxHasher;
use triomphe::{HeaderSlice, HeaderWithLength, ThinArc};

type InternMap<T> = DashMap<
    ThinArc<<T as SliceInternable>::Header, <T as SliceInternable>::SliceType>,
    (),
    BuildHasherDefault<FxHasher>,
>;
type Guard<T> = dashmap::RwLockWriteGuard<
    'static,
    RawTable<(
        ThinArc<<T as SliceInternable>::Header, <T as SliceInternable>::SliceType>,
        SharedValue<()>,
    )>,
>;
type Pointee<T> = HeaderSlice<
    HeaderWithLength<<T as SliceInternable>::Header>,
    [<T as SliceInternable>::SliceType],
>;

pub struct InternedSlice<T: SliceInternable> {
    arc: ThinArc<T::Header, T::SliceType>,
}

impl<T: SliceInternable> InternedSlice<T> {
    #[inline]
    fn new<'a>(
        header: T::Header,
        slice: impl Borrow<[T::SliceType]>
        + IntoIterator<Item = T::SliceType, IntoIter: ExactSizeIterator>,
    ) -> InternedSliceRef<'a, T> {
        const { assert!(T::USE_GC) };
        let storage = T::storage().get();
        let (mut shard, hash) = Self::select(storage, &header, slice.borrow());
        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, clone its `Arc` and return it
        //   - if not, box it up, insert it, and return a clone
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.
        let bucket = match shard.find_or_find_insert_slot(
            hash,
            |(other, _)| other.header.header == header && other.slice == *slice.borrow(),
            |(x, _)| storage.hasher().hash_one(x),
        ) {
            Ok(bucket) => bucket,
            // SAFETY: The slot came from `find_or_find_insert_slot()`, and the table wasn't modified since then.
            Err(insert_slot) => unsafe {
                shard.insert_in_slot(
                    hash,
                    insert_slot,
                    (
                        ThinArc::from_header_and_iter(header, slice.into_iter()),
                        SharedValue::new(()),
                    ),
                )
            },
        };
        // SAFETY: We just retrieved/inserted this bucket.
        unsafe {
            InternedSliceRef {
                ptr: NonNull::new_unchecked(ThinArc::as_ptr(&bucket.as_ref().0).cast_mut()),
                _marker: PhantomData,
            }
        }
    }

    #[inline]
    pub fn from_header_and_slice<'a>(
        header: T::Header,
        slice: &[T::SliceType],
    ) -> InternedSliceRef<'a, T>
    where
        T::SliceType: Clone,
    {
        return Self::new(header, Iter(slice));

        struct Iter<'a, T>(&'a [T]);

        impl<'a, T: Clone> IntoIterator for Iter<'a, T> {
            type IntoIter = std::iter::Cloned<std::slice::Iter<'a, T>>;
            type Item = T;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.0.iter().cloned()
            }
        }

        impl<T> Borrow<[T]> for Iter<'_, T> {
            #[inline]
            fn borrow(&self) -> &[T] {
                self.0
            }
        }
    }

    #[inline]
    fn select(
        storage: &'static InternMap<T>,
        header: &T::Header,
        slice: &[T::SliceType],
    ) -> (Guard<T>, u64) {
        let hash = Self::hash(storage, header, slice);
        let shard_idx = storage.determine_shard(hash as usize);
        let shard = &storage.shards()[shard_idx];
        (shard.write(), hash)
    }

    #[inline]
    fn hash(storage: &'static InternMap<T>, header: &T::Header, slice: &[T::SliceType]) -> u64 {
        storage.hasher().hash_one(HeaderSlice {
            header: HeaderWithLength { header, length: slice.len() },
            slice,
        })
    }

    #[inline(always)]
    fn ptr(&self) -> *const c_void {
        unsafe { ptr::from_ref(&self.arc).read().into_raw() }
    }

    #[inline]
    pub fn as_ref(&self) -> InternedSliceRef<'_, T> {
        InternedSliceRef {
            ptr: unsafe { NonNull::new_unchecked(self.ptr().cast_mut()) },
            _marker: PhantomData,
        }
    }
}

impl<T: SliceInternable> Drop for InternedSlice<T> {
    #[inline]
    fn drop(&mut self) {
        // When the last `Ref` is dropped, remove the object from the global map.
        if !T::USE_GC && ThinArc::strong_count(&self.arc) == 2 {
            // Only `self` and the global map point to the object.

            self.drop_slow();
        }
    }
}

impl<T: SliceInternable> InternedSlice<T> {
    #[cold]
    fn drop_slow(&mut self) {
        let storage = T::storage().get();
        let (mut shard, hash) = Self::select(storage, &self.arc.header.header, &self.arc.slice);

        if ThinArc::strong_count(&self.arc) != 2 {
            // Another thread has interned another copy
            return;
        }

        shard.remove_entry(hash, |(other, _)| **other == *self.arc);

        // Shrink the backing storage if the shard is less than 50% occupied.
        if shard.len() * 2 < shard.capacity() {
            let len = shard.len();
            shard.shrink_to(len, |(x, _)| storage.hasher().hash_one(x));
        }
    }
}

/// Compares interned `Ref`s using pointer equality.
impl<T: SliceInternable> PartialEq for InternedSlice<T> {
    // NOTE: No `?Sized` because `ptr_eq` doesn't work right with trait objects.

    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.arc.as_ptr() == other.arc.as_ptr()
    }
}

impl<T: SliceInternable> Eq for InternedSlice<T> {}

impl<T: SliceInternable> Hash for InternedSlice<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.ptr().addr())
    }
}

impl<T: SliceInternable> Deref for InternedSlice<T> {
    type Target = Pointee<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T: SliceInternable> Clone for InternedSlice<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { arc: self.arc.clone() }
    }
}

impl<T> Debug for InternedSlice<T>
where
    T: SliceInternable,
    T::SliceType: Debug,
    T::Header: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self.arc).fmt(f)
    }
}

#[repr(transparent)]
pub struct InternedSliceRef<'a, T> {
    ptr: NonNull<c_void>,
    _marker: PhantomData<&'a T>,
}

unsafe impl<T: Send + Sync> Send for InternedSliceRef<'_, T> {}
unsafe impl<T: Send + Sync> Sync for InternedSliceRef<'_, T> {}

impl<'a, T: SliceInternable> InternedSliceRef<'a, T> {
    #[inline(always)]
    fn arc(self) -> ManuallyDrop<ThinArc<T::Header, T::SliceType>> {
        unsafe { ManuallyDrop::new(ThinArc::from_raw(self.ptr.as_ptr())) }
    }

    #[inline]
    pub fn to_owned(self) -> InternedSlice<T> {
        InternedSlice { arc: (*self.arc()).clone() }
    }

    #[inline]
    pub fn get(self) -> &'a Pointee<T> {
        unsafe { &*ptr::from_ref::<Pointee<T>>(&*self.arc()) }
    }

    /// # Safety
    ///
    /// You have to make sure the data is not referenced after the refcount reaches zero; beware the interning
    /// map also keeps a reference to the value.
    #[inline]
    pub unsafe fn decrement_refcount(self) {
        drop(ManuallyDrop::into_inner(self.arc()));
    }
}

impl<T> Clone for InternedSliceRef<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for InternedSliceRef<'_, T> {}

impl<T: SliceInternable> Hash for InternedSliceRef<'_, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.ptr.as_ptr().addr());
    }
}

impl<T: SliceInternable> PartialEq for InternedSliceRef<'_, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: SliceInternable> Eq for InternedSliceRef<'_, T> {}

impl<T: SliceInternable> Deref for InternedSliceRef<'_, T> {
    type Target = Pointee<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> Debug for InternedSliceRef<'_, T>
where
    T: SliceInternable,
    T::SliceType: Debug,
    T::Header: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

pub struct InternSliceStorage<T: SliceInternable> {
    map: OnceLock<InternMap<T>>,
}

#[allow(
    clippy::new_without_default,
    reason = "this a const fn, so it can't be default yet. See <https://github.com/rust-lang/rust/issues/63065>"
)]
impl<T: SliceInternable> InternSliceStorage<T> {
    pub const fn new() -> Self {
        Self { map: OnceLock::new() }
    }
}

impl<T: SliceInternable> InternSliceStorage<T> {
    fn get(&self) -> &InternMap<T> {
        self.map.get_or_init(DashMap::default)
    }
}

pub trait SliceInternable: Sized + 'static {
    const USE_GC: bool;
    type Header: Eq + Hash;
    type SliceType: Eq + Hash + 'static;
    fn storage() -> &'static InternSliceStorage<Self>;
}

/// Implements `SliceInternable` for a given list of types, making them usable with `InternedSlice`.
#[macro_export]
#[doc(hidden)]
macro_rules! _impl_slice_internable {
    ( gc; $tag:ident, $h:ty, $t:ty $(,)? ) => {
        pub struct $tag;
        impl $crate::SliceInternable for $tag {
            const USE_GC: bool = true;
            type Header = $h;
            type SliceType = $t;
            fn storage() -> &'static $crate::InternSliceStorage<Self> {
                static STORAGE: $crate::InternSliceStorage<$tag> =
                    $crate::InternSliceStorage::new();
                &STORAGE
            }
        }
    };
}
pub use crate::_impl_slice_internable as impl_slice_internable;
