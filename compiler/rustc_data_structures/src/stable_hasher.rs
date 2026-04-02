use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::num::NonZero;

use rustc_index::bit_set::{self, DenseBitSet};
use rustc_index::{Idx, IndexSlice, IndexVec};
use smallvec::SmallVec;

#[cfg(test)]
mod tests;

use rustc_hashes::{Hash64, Hash128};
pub use rustc_stable_hash::{
    FromStableHash, SipHasher128Hash as StableHasherHash, StableSipHasher128 as StableHasher,
};

/// Something that implements `HashStable<Hcx>` can be hashed in a way that is
/// stable across multiple compilation sessions.
///
/// Note that `HashStable` imposes rather more strict requirements than usual
/// hash functions:
///
/// - Stable hashes are sometimes used as identifiers. Therefore they must
///   conform to the corresponding `PartialEq` implementations:
///
///     - `x == y` implies `hash_stable(x) == hash_stable(y)`, and
///     - `x != y` implies `hash_stable(x) != hash_stable(y)`.
///
///   That second condition is usually not required for hash functions
///   (e.g. `Hash`). In practice this means that `hash_stable` must feed any
///   information into the hasher that a `PartialEq` comparison takes into
///   account. See [#49300](https://github.com/rust-lang/rust/issues/49300)
///   for an example where violating this invariant has caused trouble in the
///   past.
///
/// - `hash_stable()` must be independent of the current
///    compilation session. E.g. they must not hash memory addresses or other
///    things that are "randomly" assigned per compilation session.
///
/// - `hash_stable()` must be independent of the host architecture. The
///   `StableHasher` takes care of endianness and `isize`/`usize` platform
///   differences.
pub trait HashStable<Hcx> {
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher);
}

/// Implement this for types that can be turned into stable keys like, for
/// example, for DefId that can be converted to a DefPathHash. This is used for
/// bringing maps into a predictable order before hashing them.
pub trait ToStableHashKey<Hcx> {
    type KeyType: Ord + Sized + HashStable<Hcx>;
    fn to_stable_hash_key(&self, hcx: &Hcx) -> Self::KeyType;
}

/// Trait for marking a type as having a sort order that is
/// stable across compilation session boundaries. More formally:
///
/// ```txt
/// Ord::cmp(a1, b1) == Ord::cmp(a2, b2)
///    where a2 = decode(encode(a1, context1), context2)
///          b2 = decode(encode(b1, context1), context2)
/// ```
///
/// i.e. the result of `Ord::cmp` is not influenced by encoding
/// the values in one session and then decoding them in another
/// session.
///
/// This is trivially true for types where encoding and decoding
/// don't change the bytes of the values that are used during
/// comparison and comparison only depends on these bytes (as
/// opposed to some non-local state). Examples are u32, String,
/// Path, etc.
///
/// But it is not true for:
///  - `*const T` and `*mut T` because the values of these pointers
///    will change between sessions.
///  - `DefIndex`, `CrateNum`, `LocalDefId`, because their concrete
///    values depend on state that might be different between
///    compilation sessions.
///
/// The associated constant `CAN_USE_UNSTABLE_SORT` denotes whether
/// unstable sorting can be used for this type. Set to true if and
/// only if `a == b` implies `a` and `b` are fully indistinguishable.
pub trait StableOrd: Ord {
    const CAN_USE_UNSTABLE_SORT: bool;

    /// Marker to ensure that implementors have carefully considered
    /// whether their `Ord` implementation obeys this trait's contract.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: ();
}

impl<T: StableOrd> StableOrd for &T {
    const CAN_USE_UNSTABLE_SORT: bool = T::CAN_USE_UNSTABLE_SORT;

    // Ordering of a reference is exactly that of the referent, and since
    // the ordering of the referet is stable so must be the ordering of the
    // reference.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

/// This is a companion trait to `StableOrd`. Some types like `Symbol` can be
/// compared in a cross-session stable way, but their `Ord` implementation is
/// not stable. In such cases, a `StableOrd` implementation can be provided
/// to offer a lightweight way for stable sorting. (The more heavyweight option
/// is to sort via `ToStableHashKey`, but then sorting needs to have access to
/// a stable hashing context and `ToStableHashKey` can also be expensive as in
/// the case of `Symbol` where it has to allocate a `String`.)
///
/// See the documentation of [StableOrd] for how stable sort order is defined.
/// The same definition applies here. Be careful when implementing this trait.
pub trait StableCompare {
    const CAN_USE_UNSTABLE_SORT: bool;

    fn stable_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

/// `StableOrd` denotes that the type's `Ord` implementation is stable, so
/// we can implement `StableCompare` by just delegating to `Ord`.
impl<T: StableOrd> StableCompare for T {
    const CAN_USE_UNSTABLE_SORT: bool = T::CAN_USE_UNSTABLE_SORT;

    fn stable_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
}

/// Implement HashStable by just calling `Hash::hash()`. Also implement `StableOrd` for the type since
/// that has the same requirements.
///
/// **WARNING** This is only valid for types that *really* don't need any context for fingerprinting.
/// But it is easy to misuse this macro (see [#96013](https://github.com/rust-lang/rust/issues/96013)
/// for examples). Therefore this macro is not exported and should only be used in the limited cases
/// here in this module.
///
/// Use `#[derive(HashStable_Generic)]` instead.
macro_rules! impl_stable_traits_for_trivial_type {
    ($t:ty) => {
        impl<Hcx> $crate::stable_hasher::HashStable<Hcx> for $t {
            #[inline]
            fn hash_stable(&self, _: &Hcx, hasher: &mut $crate::stable_hasher::StableHasher) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }

        impl $crate::stable_hasher::StableOrd for $t {
            const CAN_USE_UNSTABLE_SORT: bool = true;

            // Encoding and decoding doesn't change the bytes of trivial types
            // and `Ord::cmp` depends only on those bytes.
            const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
        }
    };
}

pub(crate) use impl_stable_traits_for_trivial_type;

impl_stable_traits_for_trivial_type!(i8);
impl_stable_traits_for_trivial_type!(i16);
impl_stable_traits_for_trivial_type!(i32);
impl_stable_traits_for_trivial_type!(i64);
impl_stable_traits_for_trivial_type!(isize);

impl_stable_traits_for_trivial_type!(u8);
impl_stable_traits_for_trivial_type!(u16);
impl_stable_traits_for_trivial_type!(u32);
impl_stable_traits_for_trivial_type!(u64);
impl_stable_traits_for_trivial_type!(usize);

impl_stable_traits_for_trivial_type!(u128);
impl_stable_traits_for_trivial_type!(i128);

impl_stable_traits_for_trivial_type!(char);
impl_stable_traits_for_trivial_type!(());

impl_stable_traits_for_trivial_type!(Hash64);

// We need a custom impl as the default hash function will only hash half the bits. For stable
// hashing we want to hash the full 128-bit hash.
impl<Hcx> HashStable<Hcx> for Hash128 {
    #[inline]
    fn hash_stable(&self, _: &Hcx, hasher: &mut StableHasher) {
        self.as_u128().hash(hasher);
    }
}

impl StableOrd for Hash128 {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // Encoding and decoding doesn't change the bytes of `Hash128`
    // and `Ord::cmp` depends only on those bytes.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<Hcx> HashStable<Hcx> for ! {
    fn hash_stable(&self, _hcx: &Hcx, _hasher: &mut StableHasher) {
        unreachable!()
    }
}

impl<Hcx, T> HashStable<Hcx> for PhantomData<T> {
    fn hash_stable(&self, _hcx: &Hcx, _hasher: &mut StableHasher) {}
}

impl<Hcx> HashStable<Hcx> for NonZero<u32> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.get().hash_stable(hcx, hasher)
    }
}

impl<Hcx> HashStable<Hcx> for NonZero<usize> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.get().hash_stable(hcx, hasher)
    }
}

impl<Hcx> HashStable<Hcx> for f32 {
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let val: u32 = self.to_bits();
        val.hash_stable(hcx, hasher);
    }
}

impl<Hcx> HashStable<Hcx> for f64 {
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let val: u64 = self.to_bits();
        val.hash_stable(hcx, hasher);
    }
}

impl<Hcx> HashStable<Hcx> for ::std::cmp::Ordering {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (*self as i8).hash_stable(hcx, hasher);
    }
}

impl<T1: HashStable<Hcx>, Hcx> HashStable<Hcx> for (T1,) {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let (ref _0,) = *self;
        _0.hash_stable(hcx, hasher);
    }
}

impl<T1: HashStable<Hcx>, T2: HashStable<Hcx>, Hcx> HashStable<Hcx> for (T1, T2) {
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1) = *self;
        _0.hash_stable(hcx, hasher);
        _1.hash_stable(hcx, hasher);
    }
}

impl<T1: StableOrd, T2: StableOrd> StableOrd for (T1, T2) {
    const CAN_USE_UNSTABLE_SORT: bool = T1::CAN_USE_UNSTABLE_SORT && T2::CAN_USE_UNSTABLE_SORT;

    // Ordering of tuples is a pure function of their elements' ordering, and since
    // the ordering of each element is stable so must be the ordering of the tuple.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2, T3, Hcx> HashStable<Hcx> for (T1, T2, T3)
where
    T1: HashStable<Hcx>,
    T2: HashStable<Hcx>,
    T3: HashStable<Hcx>,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2) = *self;
        _0.hash_stable(hcx, hasher);
        _1.hash_stable(hcx, hasher);
        _2.hash_stable(hcx, hasher);
    }
}

impl<T1: StableOrd, T2: StableOrd, T3: StableOrd> StableOrd for (T1, T2, T3) {
    const CAN_USE_UNSTABLE_SORT: bool =
        T1::CAN_USE_UNSTABLE_SORT && T2::CAN_USE_UNSTABLE_SORT && T3::CAN_USE_UNSTABLE_SORT;

    // Ordering of tuples is a pure function of their elements' ordering, and since
    // the ordering of each element is stable so must be the ordering of the tuple.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2, T3, T4, Hcx> HashStable<Hcx> for (T1, T2, T3, T4)
where
    T1: HashStable<Hcx>,
    T2: HashStable<Hcx>,
    T3: HashStable<Hcx>,
    T4: HashStable<Hcx>,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2, ref _3) = *self;
        _0.hash_stable(hcx, hasher);
        _1.hash_stable(hcx, hasher);
        _2.hash_stable(hcx, hasher);
        _3.hash_stable(hcx, hasher);
    }
}

impl<T1: StableOrd, T2: StableOrd, T3: StableOrd, T4: StableOrd> StableOrd for (T1, T2, T3, T4) {
    const CAN_USE_UNSTABLE_SORT: bool = T1::CAN_USE_UNSTABLE_SORT
        && T2::CAN_USE_UNSTABLE_SORT
        && T3::CAN_USE_UNSTABLE_SORT
        && T4::CAN_USE_UNSTABLE_SORT;

    // Ordering of tuples is a pure function of their elements' ordering, and since
    // the ordering of each element is stable so must be the ordering of the tuple.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T: HashStable<Hcx>, Hcx> HashStable<Hcx> for [T] {
    default fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for item in self {
            item.hash_stable(hcx, hasher);
        }
    }
}

impl<Hcx> HashStable<Hcx> for [u8] {
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        hasher.write(self);
    }
}

impl<T: HashStable<Hcx>, Hcx> HashStable<Hcx> for Vec<T> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self[..].hash_stable(hcx, hasher);
    }
}

impl<K, V, R, Hcx> HashStable<Hcx> for indexmap::IndexMap<K, V, R>
where
    K: HashStable<Hcx> + Eq + Hash,
    V: HashStable<Hcx>,
    R: BuildHasher,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for kv in self {
            kv.hash_stable(hcx, hasher);
        }
    }
}

impl<K, R, Hcx> HashStable<Hcx> for indexmap::IndexSet<K, R>
where
    K: HashStable<Hcx> + Eq + Hash,
    R: BuildHasher,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for key in self {
            key.hash_stable(hcx, hasher);
        }
    }
}

impl<A, const N: usize, Hcx> HashStable<Hcx> for SmallVec<[A; N]>
where
    A: HashStable<Hcx>,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self[..].hash_stable(hcx, hasher);
    }
}

impl<T: ?Sized + HashStable<Hcx>, Hcx> HashStable<Hcx> for Box<T> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher);
    }
}

impl<T: ?Sized + HashStable<Hcx>, Hcx> HashStable<Hcx> for ::std::rc::Rc<T> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher);
    }
}

impl<T: ?Sized + HashStable<Hcx>, Hcx> HashStable<Hcx> for ::std::sync::Arc<T> {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher);
    }
}

impl<Hcx> HashStable<Hcx> for str {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.as_bytes().hash_stable(hcx, hasher);
    }
}

impl StableOrd for &str {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // Encoding and decoding doesn't change the bytes of string slices
    // and `Ord::cmp` depends only on those bytes.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<Hcx> HashStable<Hcx> for String {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self[..].hash_stable(hcx, hasher);
    }
}

impl StableOrd for String {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // String comparison only depends on their contents and the
    // contents are not changed by (de-)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<Hcx> ToStableHashKey<Hcx> for String {
    type KeyType = String;
    #[inline]
    fn to_stable_hash_key(&self, _: &Hcx) -> Self::KeyType {
        self.clone()
    }
}

impl<Hcx, T1: ToStableHashKey<Hcx>, T2: ToStableHashKey<Hcx>> ToStableHashKey<Hcx> for (T1, T2) {
    type KeyType = (T1::KeyType, T2::KeyType);
    #[inline]
    fn to_stable_hash_key(&self, hcx: &Hcx) -> Self::KeyType {
        (self.0.to_stable_hash_key(hcx), self.1.to_stable_hash_key(hcx))
    }
}

impl<Hcx> HashStable<Hcx> for bool {
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (if *self { 1u8 } else { 0u8 }).hash_stable(hcx, hasher);
    }
}

impl StableOrd for bool {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // sort order of bools is not changed by (de-)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T, Hcx> HashStable<Hcx> for Option<T>
where
    T: HashStable<Hcx>,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        if let Some(ref value) = *self {
            1u8.hash_stable(hcx, hasher);
            value.hash_stable(hcx, hasher);
        } else {
            0u8.hash_stable(hcx, hasher);
        }
    }
}

impl<T: StableOrd> StableOrd for Option<T> {
    const CAN_USE_UNSTABLE_SORT: bool = T::CAN_USE_UNSTABLE_SORT;

    // the Option wrapper does not add instability to comparison.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2, Hcx> HashStable<Hcx> for Result<T1, T2>
where
    T1: HashStable<Hcx>,
    T2: HashStable<Hcx>,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            Ok(ref x) => x.hash_stable(hcx, hasher),
            Err(ref x) => x.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, T, Hcx> HashStable<Hcx> for &'a T
where
    T: HashStable<Hcx> + ?Sized,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher);
    }
}

impl<T, Hcx> HashStable<Hcx> for ::std::mem::Discriminant<T> {
    #[inline]
    fn hash_stable(&self, _: &Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<T, Hcx> HashStable<Hcx> for ::std::ops::RangeInclusive<T>
where
    T: HashStable<Hcx>,
{
    #[inline]
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.start().hash_stable(hcx, hasher);
        self.end().hash_stable(hcx, hasher);
    }
}

impl<I: Idx, T, Hcx> HashStable<Hcx> for IndexSlice<I, T>
where
    T: HashStable<Hcx>,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for v in &self.raw {
            v.hash_stable(hcx, hasher);
        }
    }
}

impl<I: Idx, T, Hcx> HashStable<Hcx> for IndexVec<I, T>
where
    T: HashStable<Hcx>,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for v in &self.raw {
            v.hash_stable(hcx, hasher);
        }
    }
}

impl<I: Idx, Hcx> HashStable<Hcx> for DenseBitSet<I> {
    fn hash_stable(&self, _hcx: &Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<R: Idx, C: Idx, Hcx> HashStable<Hcx> for bit_set::BitMatrix<R, C> {
    fn hash_stable(&self, _hcx: &Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl_stable_traits_for_trivial_type!(::std::ffi::OsStr);

impl_stable_traits_for_trivial_type!(::std::path::Path);
impl_stable_traits_for_trivial_type!(::std::path::PathBuf);

// It is not safe to implement HashStable for HashSet, HashMap or any other collection type
// with unstable but observable iteration order.
// See https://github.com/rust-lang/compiler-team/issues/533 for further information.
impl<V, Hcx> !HashStable<Hcx> for std::collections::HashSet<V> {}
impl<K, V, Hcx> !HashStable<Hcx> for std::collections::HashMap<K, V> {}

impl<K, V, Hcx> HashStable<Hcx> for ::std::collections::BTreeMap<K, V>
where
    K: HashStable<Hcx> + StableOrd,
    V: HashStable<Hcx>,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for entry in self.iter() {
            entry.hash_stable(hcx, hasher);
        }
    }
}

impl<K, Hcx> HashStable<Hcx> for ::std::collections::BTreeSet<K>
where
    K: HashStable<Hcx> + StableOrd,
{
    fn hash_stable(&self, hcx: &Hcx, hasher: &mut StableHasher) {
        self.len().hash_stable(hcx, hasher);
        for entry in self.iter() {
            entry.hash_stable(hcx, hasher);
        }
    }
}

/// Controls what data we do or do not hash.
/// Whenever a `HashStable` implementation caches its
/// result, it needs to include `HashingControls` as part
/// of the key, to ensure that it does not produce an incorrect
/// result (for example, using a `Fingerprint` produced while
/// hashing `Span`s when a `Fingerprint` without `Span`s is
/// being requested)
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct HashingControls {
    pub hash_spans: bool,
}
