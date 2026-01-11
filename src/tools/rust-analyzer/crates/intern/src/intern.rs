//! Interning of single values.
//!
//! Interning supports two modes: GC and non-GC.
//!
//! In non-GC mode, you create [`Interned`]s, and can create `Copy` handles to them
//! that can still be upgraded back to [`Interned`] ([`InternedRef`]) via [`Interned::as_ref`].
//! Generally, letting the [`InternedRef`] to outlive the [`Interned`] is a soundness bug and can
//! lead to UB. When all [`Interned`]s of some value are dropped, the value is freed (newer interns
//! may re-create it, not necessarily in the same place).
//!
//! In GC mode, you generally operate on [`InternedRef`]s. They are `Copy` and comfortable. To intern
//! a value you call [`Interned::new_gc`], which returns an [`InternedRef`]. Having all [`Interned`]s
//! of some value be dropped will *not* immediately free the value. Instead, a mark-and-sweep GC can
//! be initiated, which will free all values which have no live [`Interned`]s.
//!
//! Generally, in GC mode, you operate on [`InternedRef`], but when you need to store some long-term
//! value (e.g. a Salsa query output), you convert it to an [`Interned`]. This ensures that an eventual
//! GC will not free it as long as it is alive.
//!
//! Making mistakes is hard due to GC [`InternedRef`] wrappers not implementing `salsa::Update`, meaning
//! Salsa will ensure you do not store them in queries or Salsa-interneds. However it's still *possible*
//! without unsafe code (for example, by storing them in a `static`), which is why triggering GC is unsafe.
//!
//! For more information about GC see [`crate::gc`].

use std::{
    fmt::{self, Debug, Display},
    hash::{BuildHasher, Hash, Hasher},
    ops::Deref,
    ptr,
    sync::OnceLock,
};

use dashmap::{DashMap, SharedValue};
use hashbrown::raw::RawTable;
use rustc_hash::FxBuildHasher;
use triomphe::{Arc, ArcBorrow};

type InternMap<T> = DashMap<Arc<T>, (), FxBuildHasher>;
type Guard<T> = dashmap::RwLockWriteGuard<'static, RawTable<(Arc<T>, SharedValue<()>)>>;

pub struct Interned<T: Internable> {
    arc: Arc<T>,
}

impl<T: Internable> Interned<T> {
    #[inline]
    pub fn new(obj: T) -> Self {
        const { assert!(!T::USE_GC) };

        let storage = T::storage().get();
        let (mut shard, hash) = Self::select(storage, &obj);
        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, clone its `Arc` and return it
        //   - if not, box it up, insert it, and return a clone
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.
        let bucket = match shard.find_or_find_insert_slot(
            hash,
            |(other, _)| **other == obj,
            |(x, _)| Self::hash(storage, x),
        ) {
            Ok(bucket) => bucket,
            // SAFETY: The slot came from `find_or_find_insert_slot()`, and the table wasn't modified since then.
            Err(insert_slot) => unsafe {
                shard.insert_in_slot(hash, insert_slot, (Arc::new(obj), SharedValue::new(())))
            },
        };
        // SAFETY: We just retrieved/inserted this bucket.
        unsafe { Self { arc: bucket.as_ref().0.clone() } }
    }

    #[inline]
    pub fn new_gc<'a>(obj: T) -> InternedRef<'a, T> {
        const { assert!(T::USE_GC) };

        let storage = T::storage().get();
        let (mut shard, hash) = Self::select(storage, &obj);
        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, clone its `Arc` and return it
        //   - if not, box it up, insert it, and return a clone
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.
        let bucket = match shard.find_or_find_insert_slot(
            hash,
            |(other, _)| **other == obj,
            |(x, _)| Self::hash(storage, x),
        ) {
            Ok(bucket) => bucket,
            // SAFETY: The slot came from `find_or_find_insert_slot()`, and the table wasn't modified since then.
            Err(insert_slot) => unsafe {
                shard.insert_in_slot(hash, insert_slot, (Arc::new(obj), SharedValue::new(())))
            },
        };
        // SAFETY: We just retrieved/inserted this bucket.
        unsafe { InternedRef { arc: Arc::borrow_arc(&bucket.as_ref().0) } }
    }

    #[inline]
    fn select(storage: &'static InternMap<T>, obj: &T) -> (Guard<T>, u64) {
        let hash = Self::hash(storage, obj);
        let shard_idx = storage.determine_shard(hash as usize);
        let shard = &storage.shards()[shard_idx];
        (shard.write(), hash)
    }

    #[inline]
    fn hash(storage: &'static InternMap<T>, obj: &T) -> u64 {
        storage.hasher().hash_one(obj)
    }

    /// # Safety
    ///
    /// The pointer should originate from an `Interned` or an `InternedRef`.
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // SAFETY: Our precondition.
        Self { arc: unsafe { Arc::from_raw(ptr) } }
    }

    #[inline]
    pub fn as_ref(&self) -> InternedRef<'_, T> {
        InternedRef { arc: self.arc.borrow_arc() }
    }
}

impl<T: Internable> Drop for Interned<T> {
    #[inline]
    fn drop(&mut self) {
        // When the last `Ref` is dropped, remove the object from the global map.
        if !T::USE_GC && Arc::count(&self.arc) == 2 {
            // Only `self` and the global map point to the object.

            self.drop_slow();
        }
    }
}

impl<T: Internable> Interned<T> {
    #[cold]
    fn drop_slow(&mut self) {
        let storage = T::storage().get();
        let (mut shard, hash) = Self::select(storage, &self.arc);

        if Arc::count(&self.arc) != 2 {
            // Another thread has interned another copy
            return;
        }

        shard.remove_entry(hash, |(other, _)| **other == **self);

        // Shrink the backing storage if the shard is less than 50% occupied.
        if shard.len() * 2 < shard.capacity() {
            let len = shard.len();
            shard.shrink_to(len, |(x, _)| Self::hash(storage, x));
        }
    }
}

/// Compares interned `Ref`s using pointer equality.
impl<T: Internable> PartialEq for Interned<T> {
    // NOTE: No `?Sized` because `ptr_eq` doesn't work right with trait objects.

    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.arc, &other.arc)
    }
}

impl<T: Internable> Eq for Interned<T> {}

impl<T: Internable> Hash for Interned<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.arc.as_ptr().addr())
    }
}

impl<T: Internable> AsRef<T> for Interned<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: Internable> Deref for Interned<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T: Internable> Clone for Interned<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { arc: self.arc.clone() }
    }
}

impl<T: Debug + Internable> Debug for Interned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(&**self, f)
    }
}

impl<T: Display + Internable> Display for Interned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(&**self, f)
    }
}

#[repr(transparent)]
pub struct InternedRef<'a, T> {
    arc: ArcBorrow<'a, T>,
}

impl<'a, T: Internable> InternedRef<'a, T> {
    #[inline]
    pub fn as_raw(self) -> *const T {
        // Not `ptr::from_ref(&*self.arc)`, because we need to keep the provenance.
        self.arc.with_arc(|arc| Arc::as_ptr(arc))
    }

    /// # Safety
    ///
    /// The pointer needs to originate from `Interned` or `InternedRef`.
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // SAFETY: Our precondition.
        Self { arc: unsafe { ArcBorrow::from_ptr(ptr) } }
    }

    #[inline]
    pub fn to_owned(self) -> Interned<T> {
        Interned { arc: self.arc.clone_arc() }
    }

    #[inline]
    pub fn get(self) -> &'a T {
        self.arc.get()
    }

    /// # Safety
    ///
    /// You have to make sure the data is not referenced after the refcount reaches zero; beware the interning
    /// map also keeps a reference to the value.
    #[inline]
    pub unsafe fn decrement_refcount(self) {
        // SAFETY: Our precondition.
        unsafe { drop(Arc::from_raw(self.as_raw())) }
    }

    #[inline]
    pub(crate) fn strong_count(self) -> usize {
        ArcBorrow::strong_count(&self.arc)
    }

    /// **Available only on GC mode**.
    ///
    /// Changes the attached lifetime, as in GC mode, the lifetime is more kind of a lint to prevent misuse
    /// than actual soundness check.
    #[inline]
    pub fn change_lifetime<'b>(self) -> InternedRef<'b, T> {
        const { assert!(T::USE_GC) };
        // SAFETY: The lifetime on `InternedRef` is essentially advisory only for GCed types.
        unsafe { std::mem::transmute::<InternedRef<'a, T>, InternedRef<'b, T>>(self) }
    }
}

impl<T> Clone for InternedRef<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for InternedRef<'_, T> {}

impl<T: Hash> Hash for InternedRef<'_, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = ptr::from_ref::<T>(&*self.arc);
        state.write_usize(ptr.addr());
    }
}

impl<T: PartialEq> PartialEq for InternedRef<'_, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ArcBorrow::ptr_eq(&self.arc, &other.arc)
    }
}

impl<T: Eq> Eq for InternedRef<'_, T> {}

impl<T> Deref for InternedRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T: Debug> Debug for InternedRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self.arc).fmt(f)
    }
}

impl<T: Display> Display for InternedRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self.arc).fmt(f)
    }
}

pub struct InternStorage<T: ?Sized> {
    map: OnceLock<InternMap<T>>,
}

#[allow(
    clippy::new_without_default,
    reason = "this a const fn, so it can't be default yet. See <https://github.com/rust-lang/rust/issues/63065>"
)]
impl<T: ?Sized> InternStorage<T> {
    pub const fn new() -> Self {
        Self { map: OnceLock::new() }
    }
}

impl<T: Internable + ?Sized> InternStorage<T> {
    pub(crate) fn get(&self) -> &InternMap<T> {
        self.map.get_or_init(|| DashMap::with_capacity_and_hasher(1024, Default::default()))
    }
}

pub trait Internable: Hash + Eq + Send + Sync + 'static {
    const USE_GC: bool;

    fn storage() -> &'static InternStorage<Self>;
}

/// Implements `Internable` for a given list of types, making them usable with `Interned`.
#[macro_export]
#[doc(hidden)]
macro_rules! _impl_internable {
    ( gc; $($t:ty),+ $(,)? ) => { $(
        impl $crate::Internable for $t {
            const USE_GC: bool = true;

            fn storage() -> &'static $crate::InternStorage<Self> {
                static STORAGE: $crate::InternStorage<$t> = $crate::InternStorage::new();
                &STORAGE
            }
        }
    )+ };
    ( $($t:ty),+ $(,)? ) => { $(
        impl $crate::Internable for $t {
            const USE_GC: bool = false;

            fn storage() -> &'static $crate::InternStorage<Self> {
                static STORAGE: $crate::InternStorage<$t> = $crate::InternStorage::new();
                &STORAGE
            }
        }
    )+ };
}
pub use crate::_impl_internable as impl_internable;
