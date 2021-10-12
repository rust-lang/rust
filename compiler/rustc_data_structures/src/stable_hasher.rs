use crate::sip128::SipHasher128;
use rustc_index::bit_set;
use rustc_index::vec;
use smallvec::SmallVec;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem;

#[cfg(test)]
mod tests;

/// When hashing something that ends up affecting properties like symbol names,
/// we want these symbol names to be calculated independently of other factors
/// like what architecture you're compiling *from*.
///
/// To that end we always convert integers to little-endian format before
/// hashing and the architecture dependent `isize` and `usize` types are
/// extended to 64 bits if needed.
pub struct StableHasher {
    state: SipHasher128,
}

impl ::std::fmt::Debug for StableHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.state)
    }
}

pub trait StableHasherResult: Sized {
    fn finish(hasher: StableHasher) -> Self;
}

impl StableHasher {
    #[inline]
    pub fn new() -> Self {
        StableHasher { state: SipHasher128::new_with_keys(0, 0) }
    }

    #[inline]
    pub fn finish<W: StableHasherResult>(self) -> W {
        W::finish(self)
    }
}

impl StableHasherResult for u128 {
    fn finish(hasher: StableHasher) -> Self {
        let (_0, _1) = hasher.finalize();
        u128::from(_0) | (u128::from(_1) << 64)
    }
}

impl StableHasherResult for u64 {
    fn finish(hasher: StableHasher) -> Self {
        hasher.finalize().0
    }
}

impl StableHasher {
    #[inline]
    pub fn finalize(self) -> (u64, u64) {
        self.state.finish128()
    }
}

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        panic!("use StableHasher::finalize instead");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.state.write_u8(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.state.write_u16(i.to_le());
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.state.write_u32(i.to_le());
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state.write_u64(i.to_le());
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.state.write_u128(i.to_le());
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // Always treat usize as u64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example.
        self.state.write_u64((i as u64).to_le());
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.state.write_i16(i.to_le());
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.state.write_i32(i.to_le());
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.state.write_i64(i.to_le());
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.state.write_i128(i.to_le());
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        // Always treat isize as i64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example. Sign extending here is preferable as it means that the
        // same negative number hashes the same on both 32 and 64 bit platforms.
        self.state.write_i64((i as i64).to_le());
    }
}

/// Something that implements `HashStable<CTX>` can be hashed in a way that is
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
pub trait HashStable<CTX> {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher);
}

/// Implement this for types that can be turned into stable keys like, for
/// example, for DefId that can be converted to a DefPathHash. This is used for
/// bringing maps into a predictable order before hashing them.
pub trait ToStableHashKey<HCX> {
    type KeyType: Ord + Sized + HashStable<HCX>;
    fn to_stable_hash_key(&self, hcx: &HCX) -> Self::KeyType;
}

// Implement HashStable by just calling `Hash::hash()`. This works fine for
// self-contained values that don't depend on the hashing context `CTX`.
#[macro_export]
macro_rules! impl_stable_hash_via_hash {
    ($t:ty) => {
        impl<CTX> $crate::stable_hasher::HashStable<CTX> for $t {
            #[inline]
            fn hash_stable(&self, _: &mut CTX, hasher: &mut $crate::stable_hasher::StableHasher) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }
    };
}

impl_stable_hash_via_hash!(i8);
impl_stable_hash_via_hash!(i16);
impl_stable_hash_via_hash!(i32);
impl_stable_hash_via_hash!(i64);
impl_stable_hash_via_hash!(isize);

impl_stable_hash_via_hash!(u8);
impl_stable_hash_via_hash!(u16);
impl_stable_hash_via_hash!(u32);
impl_stable_hash_via_hash!(u64);
impl_stable_hash_via_hash!(usize);

impl_stable_hash_via_hash!(u128);
impl_stable_hash_via_hash!(i128);

impl_stable_hash_via_hash!(char);
impl_stable_hash_via_hash!(());

impl<CTX> HashStable<CTX> for ! {
    fn hash_stable(&self, _ctx: &mut CTX, _hasher: &mut StableHasher) {
        unreachable!()
    }
}

impl<CTX> HashStable<CTX> for ::std::num::NonZeroU32 {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.get().hash_stable(ctx, hasher)
    }
}

impl<CTX> HashStable<CTX> for ::std::num::NonZeroUsize {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.get().hash_stable(ctx, hasher)
    }
}

impl<CTX> HashStable<CTX> for f32 {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let val: u32 = self.to_bits();
        val.hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for f64 {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let val: u64 = self.to_bits();
        val.hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for ::std::cmp::Ordering {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (*self as i8).hash_stable(ctx, hasher);
    }
}

impl<T1: HashStable<CTX>, CTX> HashStable<CTX> for (T1,) {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let (ref _0,) = *self;
        _0.hash_stable(ctx, hasher);
    }
}

impl<T1: HashStable<CTX>, T2: HashStable<CTX>, CTX> HashStable<CTX> for (T1, T2) {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let (ref _0, ref _1) = *self;
        _0.hash_stable(ctx, hasher);
        _1.hash_stable(ctx, hasher);
    }
}

impl<T1, T2, T3, CTX> HashStable<CTX> for (T1, T2, T3)
where
    T1: HashStable<CTX>,
    T2: HashStable<CTX>,
    T3: HashStable<CTX>,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2) = *self;
        _0.hash_stable(ctx, hasher);
        _1.hash_stable(ctx, hasher);
        _2.hash_stable(ctx, hasher);
    }
}

impl<T1, T2, T3, T4, CTX> HashStable<CTX> for (T1, T2, T3, T4)
where
    T1: HashStable<CTX>,
    T2: HashStable<CTX>,
    T3: HashStable<CTX>,
    T4: HashStable<CTX>,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        let (ref _0, ref _1, ref _2, ref _3) = *self;
        _0.hash_stable(ctx, hasher);
        _1.hash_stable(ctx, hasher);
        _2.hash_stable(ctx, hasher);
        _3.hash_stable(ctx, hasher);
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for [T] {
    default fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.len().hash_stable(ctx, hasher);
        for item in self {
            item.hash_stable(ctx, hasher);
        }
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for Vec<T> {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (&self[..]).hash_stable(ctx, hasher);
    }
}

impl<K, V, R, CTX> HashStable<CTX> for indexmap::IndexMap<K, V, R>
where
    K: HashStable<CTX> + Eq + Hash,
    V: HashStable<CTX>,
    R: BuildHasher,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.len().hash_stable(ctx, hasher);
        for kv in self {
            kv.hash_stable(ctx, hasher);
        }
    }
}

impl<K, R, CTX> HashStable<CTX> for indexmap::IndexSet<K, R>
where
    K: HashStable<CTX> + Eq + Hash,
    R: BuildHasher,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.len().hash_stable(ctx, hasher);
        for key in self {
            key.hash_stable(ctx, hasher);
        }
    }
}

impl<A, CTX> HashStable<CTX> for SmallVec<[A; 1]>
where
    A: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (&self[..]).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for Box<T> {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for ::std::rc::Rc<T> {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for ::std::sync::Arc<T> {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for str {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, hasher: &mut StableHasher) {
        self.len().hash(hasher);
        self.as_bytes().hash(hasher);
    }
}

impl<CTX> HashStable<CTX> for String {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        (&self[..]).hash_stable(hcx, hasher);
    }
}

impl<HCX> ToStableHashKey<HCX> for String {
    type KeyType = String;
    #[inline]
    fn to_stable_hash_key(&self, _: &HCX) -> Self::KeyType {
        self.clone()
    }
}

impl<CTX> HashStable<CTX> for bool {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (if *self { 1u8 } else { 0u8 }).hash_stable(ctx, hasher);
    }
}

impl<T, CTX> HashStable<CTX> for Option<T>
where
    T: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        if let Some(ref value) = *self {
            1u8.hash_stable(ctx, hasher);
            value.hash_stable(ctx, hasher);
        } else {
            0u8.hash_stable(ctx, hasher);
        }
    }
}

impl<T1, T2, CTX> HashStable<CTX> for Result<T1, T2>
where
    T1: HashStable<CTX>,
    T2: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(ctx, hasher);
        match *self {
            Ok(ref x) => x.hash_stable(ctx, hasher),
            Err(ref x) => x.hash_stable(ctx, hasher),
        }
    }
}

impl<'a, T, CTX> HashStable<CTX> for &'a T
where
    T: HashStable<CTX> + ?Sized,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T, CTX> HashStable<CTX> for ::std::mem::Discriminant<T> {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, hasher: &mut StableHasher) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<T, CTX> HashStable<CTX> for ::std::ops::RangeInclusive<T>
where
    T: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.start().hash_stable(ctx, hasher);
        self.end().hash_stable(ctx, hasher);
    }
}

impl<I: vec::Idx, T, CTX> HashStable<CTX> for vec::IndexVec<I, T>
where
    T: HashStable<CTX>,
{
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.len().hash_stable(ctx, hasher);
        for v in &self.raw {
            v.hash_stable(ctx, hasher);
        }
    }
}

impl<I: vec::Idx, CTX> HashStable<CTX> for bit_set::BitSet<I> {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.words().hash_stable(ctx, hasher);
    }
}

impl<R: vec::Idx, C: vec::Idx, CTX> HashStable<CTX> for bit_set::BitMatrix<R, C> {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.words().hash_stable(ctx, hasher);
    }
}

impl<T, CTX> HashStable<CTX> for bit_set::FiniteBitSet<T>
where
    T: HashStable<CTX> + bit_set::FiniteBitSetTy,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.0.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_via_hash!(::std::path::Path);
impl_stable_hash_via_hash!(::std::path::PathBuf);

impl<K, V, R, HCX> HashStable<HCX> for ::std::collections::HashMap<K, V, R>
where
    K: ToStableHashKey<HCX> + Eq,
    V: HashStable<HCX>,
    R: BuildHasher,
{
    #[inline]
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        hash_stable_hashmap(hcx, hasher, self, ToStableHashKey::to_stable_hash_key);
    }
}

impl<K, R, HCX> HashStable<HCX> for ::std::collections::HashSet<K, R>
where
    K: ToStableHashKey<HCX> + Eq,
    R: BuildHasher,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        let mut keys: Vec<_> = self.iter().map(|k| k.to_stable_hash_key(hcx)).collect();
        keys.sort_unstable();
        keys.hash_stable(hcx, hasher);
    }
}

impl<K, V, HCX> HashStable<HCX> for ::std::collections::BTreeMap<K, V>
where
    K: ToStableHashKey<HCX>,
    V: HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        let mut entries: Vec<_> =
            self.iter().map(|(k, v)| (k.to_stable_hash_key(hcx), v)).collect();
        entries.sort_unstable_by(|&(ref sk1, _), &(ref sk2, _)| sk1.cmp(sk2));
        entries.hash_stable(hcx, hasher);
    }
}

impl<K, HCX> HashStable<HCX> for ::std::collections::BTreeSet<K>
where
    K: ToStableHashKey<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        let mut keys: Vec<_> = self.iter().map(|k| k.to_stable_hash_key(hcx)).collect();
        keys.sort_unstable();
        keys.hash_stable(hcx, hasher);
    }
}

pub fn hash_stable_hashmap<HCX, K, V, R, SK, F>(
    hcx: &mut HCX,
    hasher: &mut StableHasher,
    map: &::std::collections::HashMap<K, V, R>,
    to_stable_hash_key: F,
) where
    K: Eq,
    V: HashStable<HCX>,
    R: BuildHasher,
    SK: HashStable<HCX> + Ord,
    F: Fn(&K, &HCX) -> SK,
{
    let mut entries: Vec<_> = map.iter().map(|(k, v)| (to_stable_hash_key(k, hcx), v)).collect();
    entries.sort_unstable_by(|&(ref sk1, _), &(ref sk2, _)| sk1.cmp(sk2));
    entries.hash_stable(hcx, hasher);
}
