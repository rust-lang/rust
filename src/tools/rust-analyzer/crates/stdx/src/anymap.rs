//! This file is a port of only the necessary features from https://github.com/chris-morgan/anymap version 1.0.0-beta.2 for use within rust-analyzer.
//! Copyright © 2014–2022 Chris Morgan.
//! COPYING: https://github.com/chris-morgan/anymap/blob/master/COPYING
//! Note that the license is changed from Blue Oak Model 1.0.0 or MIT or Apache-2.0 to MIT OR Apache-2.0
//!
//! This implementation provides a safe and convenient store for one value of each type.
//!
//! Your starting point is [`Map`]. It has an example.
//!
//! # Cargo features
//!
//! This implementation has two independent features, each of which provides an implementation providing
//! types `Map`, `AnyMap`, `OccupiedEntry`, `VacantEntry`, `Entry` and `RawMap`:
//!
//! - **std** (default, *enabled* in this build):
//!   an implementation using `std::collections::hash_map`, placed in the crate root
//!   (e.g. `anymap::AnyMap`).

#![warn(missing_docs, unused_results)]

use core::hash::Hasher;

/// A hasher designed to eke a little more speed out, given `TypeId`’s known characteristics.
///
/// Specifically, this is a no-op hasher that expects to be fed a u64’s worth of
/// randomly-distributed bits. It works well for `TypeId` (eliminating start-up time, so that my
/// get_missing benchmark is ~30ns rather than ~900ns, and being a good deal faster after that, so
/// that my insert_and_get_on_260_types benchmark is ~12μs instead of ~21.5μs), but will
/// panic in debug mode and always emit zeros in release mode for any other sorts of inputs, so
/// yeah, don’t use it! 😀
#[derive(Default)]
pub struct TypeIdHasher {
    value: u64,
}

impl Hasher for TypeIdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // This expects to receive exactly one 64-bit value, and there’s no realistic chance of
        // that changing, but I don’t want to depend on something that isn’t expressly part of the
        // contract for safety. But I’m OK with release builds putting everything in one bucket
        // if it *did* change (and debug builds panicking).
        debug_assert_eq!(bytes.len(), 8);
        let _ = bytes.try_into().map(|array| self.value = u64::from_ne_bytes(array));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }
}

use core::any::{Any, TypeId};
use core::hash::BuildHasherDefault;
use core::marker::PhantomData;

use ::std::collections::hash_map;

/// Raw access to the underlying `HashMap`.
///
/// This alias is provided for convenience because of the ugly third generic parameter.
#[allow(clippy::disallowed_types)] // Uses a custom hasher
pub type RawMap<A> = hash_map::HashMap<TypeId, Box<A>, BuildHasherDefault<TypeIdHasher>>;

/// A collection containing zero or one values for any given type and allowing convenient,
/// type-safe access to those values.
///
/// The type parameter `A` allows you to use a different value type; normally you will want
/// it to be `core::any::Any` (also known as `std::any::Any`), but there are other choices:
///
/// - If you want the entire map to be cloneable, use `CloneAny` instead of `Any`; with
///   that, you can only add types that implement `Clone` to the map.
/// - You can add on `+ Send` or `+ Send + Sync` (e.g. `Map<dyn Any + Send>`) to add those
///   auto traits.
///
/// Cumulatively, there are thus six forms of map:
///
/// - <code>[Map]&lt;dyn [core::any::Any]&gt;</code>,
///   also spelled [`AnyMap`] for convenience.
/// - <code>[Map]&lt;dyn [core::any::Any] + Send&gt;</code>
/// - <code>[Map]&lt;dyn [core::any::Any] + Send + Sync&gt;</code>
/// - <code>[Map]&lt;dyn [CloneAny]&gt;</code>
/// - <code>[Map]&lt;dyn [CloneAny] + Send&gt;</code>
/// - <code>[Map]&lt;dyn [CloneAny] + Send + Sync&gt;</code>
///
/// ## Example
///
/// (Here using the [`AnyMap`] convenience alias; the first line could use
/// <code>[anymap::Map][Map]::&lt;[core::any::Any]&gt;::new()</code> instead if desired.)
///
/// ```rust
#[doc = "let mut data = anymap::AnyMap::new();"]
/// assert_eq!(data.get(), None::<&i32>);
/// ```
///
/// Values containing non-static references are not permitted.
#[derive(Debug)]
pub struct Map<A: ?Sized + Downcast = dyn Any> {
    raw: RawMap<A>,
}

/// The most common type of `Map`: just using `Any`; <code>[Map]&lt;dyn [Any]&gt;</code>.
///
/// Why is this a separate type alias rather than a default value for `Map<A>`?
/// `Map::new()` doesn’t seem to be happy to infer that it should go with the default
/// value. It’s a bit sad, really. Ah well, I guess this approach will do.
pub type AnyMap = Map<dyn Any>;
impl<A: ?Sized + Downcast> Default for Map<A> {
    #[inline]
    fn default() -> Map<A> {
        Map::new()
    }
}

impl<A: ?Sized + Downcast> Map<A> {
    /// Create an empty collection.
    #[inline]
    pub fn new() -> Map<A> {
        Map { raw: RawMap::with_hasher(Default::default()) }
    }

    /// Returns a reference to the value stored in the collection for the type `T`,
    /// if it exists.
    #[inline]
    pub fn get<T: IntoBox<A>>(&self) -> Option<&T> {
        self.raw.get(&TypeId::of::<T>()).map(|any| unsafe { any.downcast_ref_unchecked::<T>() })
    }

    /// Gets the entry for the given type in the collection for in-place manipulation
    #[inline]
    pub fn entry<T: IntoBox<A>>(&mut self) -> Entry<'_, A, T> {
        match self.raw.entry(TypeId::of::<T>()) {
            hash_map::Entry::Occupied(e) => {
                Entry::Occupied(OccupiedEntry { inner: e, type_: PhantomData })
            }
            hash_map::Entry::Vacant(e) => {
                Entry::Vacant(VacantEntry { inner: e, type_: PhantomData })
            }
        }
    }
}

/// A view into a single occupied location in an `Map`.
pub struct OccupiedEntry<'a, A: ?Sized + Downcast, V: 'a> {
    inner: hash_map::OccupiedEntry<'a, TypeId, Box<A>>,
    type_: PhantomData<V>,
}

/// A view into a single empty location in an `Map`.
pub struct VacantEntry<'a, A: ?Sized + Downcast, V: 'a> {
    inner: hash_map::VacantEntry<'a, TypeId, Box<A>>,
    type_: PhantomData<V>,
}

/// A view into a single location in an `Map`, which may be vacant or occupied.
pub enum Entry<'a, A: ?Sized + Downcast, V> {
    /// An occupied Entry
    Occupied(OccupiedEntry<'a, A, V>),
    /// A vacant Entry
    Vacant(VacantEntry<'a, A, V>),
}

impl<'a, A: ?Sized + Downcast, V: IntoBox<A>> Entry<'a, A, V> {
    /// Ensures a value is in the entry by inserting the result of the default function if
    /// empty, and returns a mutable reference to the value in the entry.
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(inner) => inner.into_mut(),
            Entry::Vacant(inner) => inner.insert(default()),
        }
    }
}

impl<'a, A: ?Sized + Downcast, V: IntoBox<A>> OccupiedEntry<'a, A, V> {
    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the collection itself
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        unsafe { self.inner.into_mut().downcast_mut_unchecked() }
    }
}

impl<'a, A: ?Sized + Downcast, V: IntoBox<A>> VacantEntry<'a, A, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        unsafe { self.inner.insert(value.into_box()).downcast_mut_unchecked() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct A(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct B(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct C(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct D(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct E(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct F(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct J(i32);

    #[test]
    fn test_varieties() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn assert_debug<T: ::core::fmt::Debug>() {}
        assert_send::<Map<dyn Any + Send>>();
        assert_send::<Map<dyn Any + Send + Sync>>();
        assert_sync::<Map<dyn Any + Send + Sync>>();
        assert_debug::<Map<dyn Any>>();
        assert_debug::<Map<dyn Any + Send>>();
        assert_debug::<Map<dyn Any + Send + Sync>>();
        assert_send::<Map<dyn CloneAny + Send>>();
        assert_send::<Map<dyn CloneAny + Send + Sync>>();
        assert_sync::<Map<dyn CloneAny + Send + Sync>>();
        assert_debug::<Map<dyn CloneAny>>();
        assert_debug::<Map<dyn CloneAny + Send>>();
        assert_debug::<Map<dyn CloneAny + Send + Sync>>();
    }

    #[test]
    fn type_id_hasher() {
        use core::any::TypeId;
        use core::hash::Hash;
        fn verify_hashing_with(type_id: TypeId) {
            let mut hasher = TypeIdHasher::default();
            type_id.hash(&mut hasher);
            // SAFETY: u64 is valid for all bit patterns.
            let _ = hasher.finish();
        }
        // Pick a variety of types, just to demonstrate it’s all sane. Normal, zero-sized, unsized, &c.
        verify_hashing_with(TypeId::of::<usize>());
        verify_hashing_with(TypeId::of::<()>());
        verify_hashing_with(TypeId::of::<str>());
        verify_hashing_with(TypeId::of::<&str>());
        verify_hashing_with(TypeId::of::<Vec<u8>>());
    }
}

// impl some traits for dyn Any
use core::fmt;

#[doc(hidden)]
pub trait CloneToAny {
    /// Clone `self` into a new `Box<dyn CloneAny>` object.
    fn clone_to_any(&self) -> Box<dyn CloneAny>;
}

impl<T: Any + Clone> CloneToAny for T {
    #[inline]
    fn clone_to_any(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }
}

macro_rules! impl_clone {
    ($t:ty) => {
        impl Clone for Box<$t> {
            #[inline]
            fn clone(&self) -> Box<$t> {
                // SAFETY: this dance is to reapply any Send/Sync marker. I’m not happy about this
                // approach, given that I used to do it in safe code, but then came a dodgy
                // future-compatibility warning where_clauses_object_safety, which is spurious for
                // auto traits but still super annoying (future-compatibility lints seem to mean
                // your bin crate needs a corresponding allow!). Although I explained my plight¹
                // and it was all explained and agreed upon, no action has been taken. So I finally
                // caved and worked around it by doing it this way, which matches what’s done for
                // core::any², so it’s probably not *too* bad.
                //
                // ¹ https://github.com/rust-lang/rust/issues/51443#issuecomment-421988013
                // ² https://github.com/rust-lang/rust/blob/e7825f2b690c9a0d21b6f6d84c404bb53b151b38/library/alloc/src/boxed.rs#L1613-L1616
                let clone: Box<dyn CloneAny> = (**self).clone_to_any();
                let raw: *mut dyn CloneAny = Box::into_raw(clone);
                unsafe { Box::from_raw(raw as *mut $t) }
            }
        }

        impl fmt::Debug for $t {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.pad(stringify!($t))
            }
        }
    };
}

/// Methods for downcasting from an `Any`-like trait object.
///
/// This should only be implemented on trait objects for subtraits of `Any`, though you can
/// implement it for other types and it’ll work fine, so long as your implementation is correct.
pub trait Downcast {
    /// Gets the `TypeId` of `self`.
    fn type_id(&self) -> TypeId;

    // Note the bound through these downcast methods is 'static, rather than the inexpressible
    // concept of Self-but-as-a-trait (where Self is `dyn Trait`). This is sufficient, exceeding
    // TypeId’s requirements. Sure, you *can* do CloneAny.downcast_unchecked::<NotClone>() and the
    // type system won’t protect you, but that doesn’t introduce any unsafety: the method is
    // already unsafe because you can specify the wrong type, and if this were exposing safe
    // downcasting, CloneAny.downcast::<NotClone>() would just return an error, which is just as
    // correct.
    //
    // Now in theory we could also add T: ?Sized, but that doesn’t play nicely with the common
    // implementation, so I’m doing without it.

    /// Downcast from `&Any` to `&T`, without checking the type matches.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `T` matches the trait object, on pain of *undefined behaviour*.
    unsafe fn downcast_ref_unchecked<T: 'static>(&self) -> &T;

    /// Downcast from `&mut Any` to `&mut T`, without checking the type matches.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `T` matches the trait object, on pain of *undefined behaviour*.
    unsafe fn downcast_mut_unchecked<T: 'static>(&mut self) -> &mut T;
}

/// A trait for the conversion of an object into a boxed trait object.
pub trait IntoBox<A: ?Sized + Downcast>: Any {
    /// Convert self into the appropriate boxed form.
    fn into_box(self) -> Box<A>;
}

macro_rules! implement {
    ($any_trait:ident $(+ $auto_traits:ident)*) => {
        impl Downcast for dyn $any_trait $(+ $auto_traits)* {
            #[inline]
            fn type_id(&self) -> TypeId {
                self.type_id()
            }

            #[inline]
            unsafe fn downcast_ref_unchecked<T: 'static>(&self) -> &T {
                &*(self as *const Self as *const T)
            }

            #[inline]
            unsafe fn downcast_mut_unchecked<T: 'static>(&mut self) -> &mut T {
                &mut *(self as *mut Self as *mut T)
            }
        }

        impl<T: $any_trait $(+ $auto_traits)*> IntoBox<dyn $any_trait $(+ $auto_traits)*> for T {
            #[inline]
            fn into_box(self) -> Box<dyn $any_trait $(+ $auto_traits)*> {
                Box::new(self)
            }
        }
    }
}

implement!(Any);
implement!(Any + Send);
implement!(Any + Send + Sync);

/// [`Any`], but with cloning.
///
/// Every type with no non-`'static` references that implements `Clone` implements `CloneAny`.
/// See [`core::any`] for more details on `Any` in general.
pub trait CloneAny: Any + CloneToAny {}
impl<T: Any + Clone> CloneAny for T {}
implement!(CloneAny);
implement!(CloneAny + Send);
implement!(CloneAny + Send + Sync);
impl_clone!(dyn CloneAny);
impl_clone!(dyn CloneAny + Send);
impl_clone!(dyn CloneAny + Send + Sync);
