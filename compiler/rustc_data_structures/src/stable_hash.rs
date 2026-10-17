use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::num::NonZero;

use rustc_index::bit_set::{self, DenseBitSet};
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_macros::StableHash;
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::fingerprint::Fingerprint;

#[cfg(test)]
mod tests;

use rustc_hashes::{Hash64, Hash128};
pub use rustc_stable_hash::{
    FromStableHash, SipHasher128Hash as StableHasherHash, StableSipHasher128 as StableHasher,
};

/// This trait lets `StableHash` and `derive(StableHash)` be used in
/// this crate (and other crates upstream of `rustc_middle`), while leaving
/// certain operations to be defined in `rustc_middle` where more things are
/// visible.
pub trait StableHashCtxt {
    /// The main event: stable hashing of a span.
    fn stable_hash_span(&mut self, span: RawSpan, hasher: &mut StableHasher);

    /// Compute a `Fingerprint`, which can be trivially turned into a `DefPathHash`.
    fn def_path_hash(&self, def_id: RawDefId) -> Fingerprint;

    /// Get the stable hash controls.
    fn stable_hash_controls(&self) -> StableHashControls;

    /// Assert that the provided `StableHashCtxt` is configured with the default
    /// `StableHashControls`. We should always have bailed out before getting to here with a
    fn assert_default_stable_hash_controls(&self, msg: &str);
}

/// Dummy impl of [`StableHashCtxt`], for code that can't provide a real context,
/// and only wants to compute stable hashes for types that don't rely on the context.
pub struct DummyStableHashCtxt;

impl StableHashCtxt for DummyStableHashCtxt {
    fn stable_hash_span(&mut self, _span: RawSpan, _hasher: &mut StableHasher) {
        panic!("method on dummy stable-hash context was unexpectedly called");
    }
    fn def_path_hash(&self, _def_id: RawDefId) -> Fingerprint {
        panic!("method on dummy stable-hash context was unexpectedly called");
    }
    fn stable_hash_controls(&self) -> StableHashControls {
        panic!("method on dummy stable-hash context was unexpectedly called");
    }
    fn assert_default_stable_hash_controls(&self, _msg: &str) {
        panic!("method on dummy stable-hash context was unexpectedly called");
    }
}

// A type used to work around `Span` not being visible in this crate. It is the same size as
// `Span` but has a different alignment.
pub struct RawSpan(pub u64);

// A type used to work around `DefId` not being visible in this crate. It is the same size as
// `DefId`.
pub struct RawDefId(pub u32, pub u32);

/// Something that implements `StableHash` can be hashed in a way that is
/// stable across multiple compilation sessions.
///
/// Note that `StableHash` imposes rather more strict requirements than usual
/// hash functions:
///
/// - Stable hashes are sometimes used as identifiers. Therefore they must
///   conform to the corresponding `PartialEq` implementations:
///
///     - `x == y` implies `stable_hash(x) == stable_hash(y)`, and
///     - `x != y` implies `stable_hash(x) != stable_hash(y)`.
///
///   That second condition is usually not required for hash functions
///   (e.g. `Hash`). In practice this means that `stable_hash` must feed any
///   information into the hasher that a `PartialEq` comparison takes into
///   account. See [#49300](https://github.com/rust-lang/rust/issues/49300)
///   for an example where violating this invariant has caused trouble in the
///   past.
///
/// - `stable_hash()` must be independent of the current
///    compilation session. E.g. they must not hash memory addresses or other
///    things that are "randomly" assigned per compilation session.
///
/// - `stable_hash()` must be independent of the host architecture. The
///   `StableHasher` takes care of endianness and `isize`/`usize` platform
///   differences.
///
/// # Consistency across compiler stages
///
/// Because the stage1 compiler is built against the bootstrap stdlib, but the stage2
/// compiler uses in-tree stdlib, implementations of `StableHash` must not rely on `Hash`
/// impls that were provided or derived by the standard library. Doing so leads to weird
/// bugs and CI failures that can block changes to those stdlib hash impls.
///
/// (Unfortunately, as of late 2026 we have many stable-hash impls that _do_
/// leak details of stdlib hash impls.)
///
/// See <https://github.com/rust-lang/rust/pull/164458> for more context.
pub trait StableHash {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher);
}

/// Implement this for types that can be turned into stable keys like, for
/// example, for DefId that can be converted to a DefPathHash. This is used for
/// bringing maps into a predictable order before hashing them.
pub trait ToStableHashKey {
    type KeyType: Ord + Sized + StableHash;
    fn to_stable_hash_key<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx) -> Self::KeyType;
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

/// Implement StableHash by just calling `Hash::hash()`. Also implement `StableOrd` for the type
/// since that has the same requirements.
///
/// **WARNING** This is only valid for types that *really* don't need any context for fingerprinting.
/// But it is easy to misuse this macro (see [#96013](https://github.com/rust-lang/rust/issues/96013)
/// for examples). Therefore this macro is not exported and should only be used in the limited cases
/// here in this module.
///
/// Use `#[derive(StableHash)]` instead.
macro_rules! impl_stable_traits_for_trivial_type {
    ($t:ty) => {
        impl $crate::stable_hash::StableHash for $t {
            #[inline]
            fn stable_hash<Hcx>(
                &self,
                _: &mut Hcx,
                hasher: &mut $crate::stable_hash::StableHasher,
            ) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }

        impl $crate::stable_hash::StableOrd for $t {
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
impl StableHash for Hash128 {
    #[inline]
    fn stable_hash<Hcx>(&self, _: &mut Hcx, hasher: &mut StableHasher) {
        self.as_u128().hash(hasher);
    }
}

impl StableOrd for Hash128 {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // Encoding and decoding doesn't change the bytes of `Hash128`
    // and `Ord::cmp` depends only on those bytes.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl StableHash for ! {
    fn stable_hash<Hcx>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {
        unreachable!()
    }
}

impl<T> StableHash for PhantomData<T> {
    fn stable_hash<Hcx>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {}
}

impl<T: StableHash + std::num::ZeroablePrimitive> StableHash for NonZero<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.get().stable_hash(hcx, hasher)
    }
}

impl StableHash for f32 {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let val: u32 = self.to_bits();
        val.stable_hash(hcx, hasher);
    }
}

impl StableHash for f64 {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let val: u64 = self.to_bits();
        val.stable_hash(hcx, hasher);
    }
}

impl StableHash for ::std::cmp::Ordering {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (*self as i8).stable_hash(hcx, hasher);
    }
}

impl<T1: StableHash> StableHash for (T1,) {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let (ref _0,) = *self;
        _0.stable_hash(hcx, hasher);
    }
}

impl<T1: StableHash, T2: StableHash> StableHash for (T1, T2) {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1) = *self;
        _0.stable_hash(hcx, hasher);
        _1.stable_hash(hcx, hasher);
    }
}

impl<T1: StableOrd, T2: StableOrd> StableOrd for (T1, T2) {
    const CAN_USE_UNSTABLE_SORT: bool = T1::CAN_USE_UNSTABLE_SORT && T2::CAN_USE_UNSTABLE_SORT;

    // Ordering of tuples is a pure function of their elements' ordering, and since
    // the ordering of each element is stable so must be the ordering of the tuple.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2, T3> StableHash for (T1, T2, T3)
where
    T1: StableHash,
    T2: StableHash,
    T3: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2) = *self;
        _0.stable_hash(hcx, hasher);
        _1.stable_hash(hcx, hasher);
        _2.stable_hash(hcx, hasher);
    }
}

impl<T1: StableOrd, T2: StableOrd, T3: StableOrd> StableOrd for (T1, T2, T3) {
    const CAN_USE_UNSTABLE_SORT: bool =
        T1::CAN_USE_UNSTABLE_SORT && T2::CAN_USE_UNSTABLE_SORT && T3::CAN_USE_UNSTABLE_SORT;

    // Ordering of tuples is a pure function of their elements' ordering, and since
    // the ordering of each element is stable so must be the ordering of the tuple.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2, T3, T4> StableHash for (T1, T2, T3, T4)
where
    T1: StableHash,
    T2: StableHash,
    T3: StableHash,
    T4: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2, ref _3) = *self;
        _0.stable_hash(hcx, hasher);
        _1.stable_hash(hcx, hasher);
        _2.stable_hash(hcx, hasher);
        _3.stable_hash(hcx, hasher);
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

impl<T: StableHash> StableHash for [T] {
    default fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for item in self {
            item.stable_hash(hcx, hasher);
        }
    }
}

impl StableHash for [u8] {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        hasher.write(self);
    }
}

impl<T: StableHash> StableHash for Vec<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self[..].stable_hash(hcx, hasher);
    }
}

impl<K, V, R> StableHash for indexmap::IndexMap<K, V, R>
where
    K: StableHash + Eq + Hash,
    V: StableHash,
    R: BuildHasher,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for kv in self {
            kv.stable_hash(hcx, hasher);
        }
    }
}

impl<K, R> StableHash for indexmap::IndexSet<K, R>
where
    K: StableHash + Eq + Hash,
    R: BuildHasher,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for key in self {
            key.stable_hash(hcx, hasher);
        }
    }
}

impl<A, const N: usize> StableHash for SmallVec<[A; N]>
where
    A: StableHash,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self[..].stable_hash(hcx, hasher);
    }
}

impl<T: StableHash> StableHash for ThinVec<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self[..].stable_hash(hcx, hasher);
    }
}

impl<T: ?Sized + StableHash> StableHash for Box<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (**self).stable_hash(hcx, hasher);
    }
}

impl<T: ?Sized + StableHash> StableHash for ::std::rc::Rc<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (**self).stable_hash(hcx, hasher);
    }
}

impl<T: ?Sized + StableHash> StableHash for ::std::sync::Arc<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (**self).stable_hash(hcx, hasher);
    }
}

impl StableHash for str {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.as_bytes().stable_hash(hcx, hasher);
    }
}

impl StableOrd for &str {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // Encoding and decoding doesn't change the bytes of string slices
    // and `Ord::cmp` depends only on those bytes.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl StableHash for String {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self[..].stable_hash(hcx, hasher);
    }
}

impl StableOrd for String {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // String comparison only depends on their contents and the
    // contents are not changed by (de-)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl StableHash for bool {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (if *self { 1u8 } else { 0u8 }).stable_hash(hcx, hasher);
    }
}

impl StableOrd for bool {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // sort order of bools is not changed by (de-)serialization.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T> StableHash for Option<T>
where
    T: StableHash,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        if let Some(ref value) = *self {
            1u8.stable_hash(hcx, hasher);
            value.stable_hash(hcx, hasher);
        } else {
            0u8.stable_hash(hcx, hasher);
        }
    }
}

impl<T: StableOrd> StableOrd for Option<T> {
    const CAN_USE_UNSTABLE_SORT: bool = T::CAN_USE_UNSTABLE_SORT;

    // the Option wrapper does not add instability to comparison.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl<T1, T2> StableHash for Result<T1, T2>
where
    T1: StableHash,
    T2: StableHash,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        mem::discriminant(self).stable_hash(hcx, hasher);
        match *self {
            Ok(ref x) => x.stable_hash(hcx, hasher),
            Err(ref x) => x.stable_hash(hcx, hasher),
        }
    }
}

impl<'a, T> StableHash for &'a T
where
    T: StableHash + ?Sized,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        (**self).stable_hash(hcx, hasher);
    }
}

impl<T> StableHash for ::std::mem::Discriminant<T> {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, _: &mut Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<T> StableHash for ::std::range::RangeInclusive<T>
where
    T: StableHash,
{
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.start.stable_hash(hcx, hasher);
        self.last.stable_hash(hcx, hasher);
    }
}

impl<I: Idx, T> StableHash for IndexSlice<I, T>
where
    T: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for v in &self.raw {
            v.stable_hash(hcx, hasher);
        }
    }
}

impl<I: Idx, T> StableHash for IndexVec<I, T>
where
    T: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for v in &self.raw {
            v.stable_hash(hcx, hasher);
        }
    }
}

impl<I: Idx> StableHash for DenseBitSet<I> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<R: Idx, C: Idx> StableHash for bit_set::BitMatrix<R, C> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl_stable_traits_for_trivial_type!(::std::ffi::OsStr);

impl StableHash for std::path::Path {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        use std::path::{Component, Prefix};

        /// Tag value to be mixed in with each kind of component/prefix.
        #[derive(StableHash)]
        enum Tag {
            StartOfPath,
            PrefixVerbatim,
            PrefixVerbatimUNC,
            PrefixVerbatimDisk,
            PrefixDeviceNS,
            PrefixUNC,
            PrefixDisk,
            RootDir,
            CurDir,
            ParentDir,
            Normal,
            EndOfPath,
        }

        // Explicitly tag the start of each path, so that empty paths always hash something.
        Tag::StartOfPath.stable_hash(hcx, hasher);

        for (i, component) in self.components().enumerate() {
            // Mix in the index of each component, to avoid potentially trivial collisions
            // between inputs like `("foo/bar", "baz")` and `("foo", "bar/baz")`.
            i.stable_hash(hcx, hasher);
            match component {
                Component::Prefix(prefix) => match prefix.kind() {
                    Prefix::Verbatim(os_str) => {
                        Tag::PrefixVerbatim.stable_hash(hcx, hasher);
                        os_str.stable_hash(hcx, hasher);
                    }
                    Prefix::VerbatimUNC(host, share) => {
                        Tag::PrefixVerbatimUNC.stable_hash(hcx, hasher);
                        host.stable_hash(hcx, hasher);
                        share.stable_hash(hcx, hasher);
                    }
                    Prefix::VerbatimDisk(letter) => {
                        Tag::PrefixVerbatimDisk.stable_hash(hcx, hasher);
                        letter.stable_hash(hcx, hasher);
                    }
                    Prefix::DeviceNS(os_str) => {
                        Tag::PrefixDeviceNS.stable_hash(hcx, hasher);
                        os_str.stable_hash(hcx, hasher);
                    }
                    Prefix::UNC(host, share) => {
                        Tag::PrefixUNC.stable_hash(hcx, hasher);
                        host.stable_hash(hcx, hasher);
                        share.stable_hash(hcx, hasher);
                    }
                    Prefix::Disk(letter) => {
                        Tag::PrefixDisk.stable_hash(hcx, hasher);
                        letter.stable_hash(hcx, hasher);
                    }
                },
                Component::RootDir => Tag::RootDir.stable_hash(hcx, hasher),
                Component::CurDir => Tag::CurDir.stable_hash(hcx, hasher),
                Component::ParentDir => Tag::ParentDir.stable_hash(hcx, hasher),
                Component::Normal(os_str) => {
                    Tag::Normal.stable_hash(hcx, hasher);
                    os_str.stable_hash(hcx, hasher);
                }
            }
        }

        // Explicitly tag the end of each path, so that empty paths always hash something.
        // (Tagging the start and the end and the component-index is potentially overkill.)
        Tag::EndOfPath.stable_hash(hcx, hasher);
    }
}

impl StableOrd for std::path::Path {
    /// FIXME(Zalathar): Unstably sorting paths is pre-existing behavior, but potentially wrong.
    const CAN_USE_UNSTABLE_SORT: bool = true;
    /// Stdlib path sorting should (hopefully) not change.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl StableHash for std::path::PathBuf {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.as_path().stable_hash(hcx, hasher);
    }
}

impl StableOrd for std::path::PathBuf {
    const CAN_USE_UNSTABLE_SORT: bool = <std::path::Path as StableOrd>::CAN_USE_UNSTABLE_SORT;
    /// Stdlib path sorting should (hopefully) not change.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

// It is not safe to implement StableHash for HashSet, HashMap or any other collection type
// with unstable but observable iteration order.
// See https://github.com/rust-lang/compiler-team/issues/533 for further information.
impl<V> !StableHash for std::collections::HashSet<V> {}
impl<K, V> !StableHash for std::collections::HashMap<K, V> {}

impl<K, V> StableHash for ::std::collections::BTreeMap<K, V>
where
    K: StableHash + StableOrd,
    V: StableHash,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for entry in self.iter() {
            entry.stable_hash(hcx, hasher);
        }
    }
}

impl<K> StableHash for ::std::collections::BTreeSet<K>
where
    K: StableHash + StableOrd,
{
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.len().stable_hash(hcx, hasher);
        for entry in self.iter() {
            entry.stable_hash(hcx, hasher);
        }
    }
}

/// Controls what data we do or do not hash.
/// Whenever a `StableHash` implementation caches its
/// result, it needs to include `StableHashControls` as part
/// of the key, to ensure that it does not produce an incorrect
/// result (for example, using a `Fingerprint` produced while
/// hashing `Span`s when a `Fingerprint` without `Span`s is
/// being requested)
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct StableHashControls {
    pub hash_spans: bool,
}
