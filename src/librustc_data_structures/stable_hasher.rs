// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::{Hash, Hasher, BuildHasher};
use std::marker::PhantomData;
use std::mem;
use sip128::SipHasher128;

/// When hashing something that ends up affecting properties like symbol names,
/// we want these symbol names to be calculated independently of other factors
/// like what architecture you're compiling *from*.
///
/// To that end we always convert integers to little-endian format before
/// hashing and the architecture dependent `isize` and `usize` types are
/// extended to 64 bits if needed.
pub struct StableHasher<W> {
    state: SipHasher128,
    bytes_hashed: u64,
    width: PhantomData<W>,
}

impl<W: StableHasherResult> ::std::fmt::Debug for StableHasher<W> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{:?}", self.state)
    }
}

pub trait StableHasherResult: Sized {
    fn finish(hasher: StableHasher<Self>) -> Self;
}

impl<W: StableHasherResult> StableHasher<W> {
    pub fn new() -> Self {
        StableHasher {
            state: SipHasher128::new_with_keys(0, 0),
            bytes_hashed: 0,
            width: PhantomData,
        }
    }

    pub fn finish(self) -> W {
        W::finish(self)
    }
}

impl StableHasherResult for u128 {
    fn finish(hasher: StableHasher<Self>) -> Self {
        let (_0, _1) = hasher.finalize();
        (_0 as u128) | ((_1 as u128) << 64)
    }
}

impl StableHasherResult for u64 {
    fn finish(hasher: StableHasher<Self>) -> Self {
        hasher.finalize().0
    }
}

impl<W> StableHasher<W> {
    #[inline]
    pub fn finalize(self) -> (u64, u64) {
        self.state.finish128()
    }

    #[inline]
    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }
}

impl<W> Hasher for StableHasher<W> {
    fn finish(&self) -> u64 {
        panic!("use StableHasher::finalize instead");
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
        self.bytes_hashed += bytes.len() as u64;
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.state.write_u8(i);
        self.bytes_hashed += 1;
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.state.write_u16(i.to_le());
        self.bytes_hashed += 2;
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.state.write_u32(i.to_le());
        self.bytes_hashed += 4;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state.write_u64(i.to_le());
        self.bytes_hashed += 8;
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.state.write_u128(i.to_le());
        self.bytes_hashed += 16;
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // Always treat usize as u64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example.
        self.state.write_u64((i as u64).to_le());
        self.bytes_hashed += 8;
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
        self.bytes_hashed += 1;
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.state.write_i16(i.to_le());
        self.bytes_hashed += 2;
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.state.write_i32(i.to_le());
        self.bytes_hashed += 4;
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.state.write_i64(i.to_le());
        self.bytes_hashed += 8;
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.state.write_i128(i.to_le());
        self.bytes_hashed += 16;
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        // Always treat isize as i64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example.
        self.state.write_i64((i as i64).to_le());
        self.bytes_hashed += 8;
    }
}

/// Something that implements `HashStable<CTX>` can be hashed in a way that is
/// stable across multiple compilation sessions.
pub trait HashStable<CTX> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>);
}

/// Implement this for types that can be turned into stable keys like, for
/// example, for DefId that can be converted to a DefPathHash. This is used for
/// bringing maps into a predictable order before hashing them.
pub trait ToStableHashKey<HCX> {
    type KeyType: Ord + Clone + Sized + HashStable<HCX>;
    fn to_stable_hash_key(&self, hcx: &HCX) -> Self::KeyType;
}

// Implement HashStable by just calling `Hash::hash()`. This works fine for
// self-contained values that don't depend on the hashing context `CTX`.
#[macro_export]
macro_rules! impl_stable_hash_via_hash {
    ($t:ty) => (
        impl<CTX> $crate::stable_hasher::HashStable<CTX> for $t {
            #[inline]
            fn hash_stable<W: $crate::stable_hasher::StableHasherResult>(
                &self,
                _: &mut CTX,
                hasher: &mut $crate::stable_hasher::StableHasher<W>
            ) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }
    );
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

impl<CTX> HashStable<CTX> for f32 {
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let val: u32 = unsafe {
            ::std::mem::transmute(*self)
        };
        val.hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for f64 {
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let val: u64 = unsafe {
            ::std::mem::transmute(*self)
        };
        val.hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for ::std::cmp::Ordering {
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (*self as i8).hash_stable(ctx, hasher);
    }
}

impl<T1: HashStable<CTX>, CTX> HashStable<CTX> for (T1,) {
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let (ref _0,) = *self;
        _0.hash_stable(ctx, hasher);
    }
}

impl<T1: HashStable<CTX>, T2: HashStable<CTX>, CTX> HashStable<CTX> for (T1, T2) {
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let (ref _0, ref _1) = *self;
        _0.hash_stable(ctx, hasher);
        _1.hash_stable(ctx, hasher);
    }
}

impl<T1, T2, T3, CTX> HashStable<CTX> for (T1, T2, T3)
     where T1: HashStable<CTX>,
           T2: HashStable<CTX>,
           T3: HashStable<CTX>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        let (ref _0, ref _1, ref _2) = *self;
        _0.hash_stable(ctx, hasher);
        _1.hash_stable(ctx, hasher);
        _2.hash_stable(ctx, hasher);
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for [T] {
    default fn hash_stable<W: StableHasherResult>(&self,
                                                  ctx: &mut CTX,
                                                  hasher: &mut StableHasher<W>) {
        self.len().hash_stable(ctx, hasher);
        for item in self {
            item.hash_stable(ctx, hasher);
        }
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for Vec<T> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (&self[..]).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for Box<T> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for ::std::rc::Rc<T> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T: ?Sized + HashStable<CTX>, CTX> HashStable<CTX> for ::std::sync::Arc<T> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<CTX> HashStable<CTX> for str {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        self.len().hash(hasher);
        self.as_bytes().hash(hasher);
    }
}


impl<CTX> HashStable<CTX> for String {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
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
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (if *self { 1u8 } else { 0u8 }).hash_stable(ctx, hasher);
    }
}


impl<T, CTX> HashStable<CTX> for Option<T>
    where T: HashStable<CTX>
{
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        if let Some(ref value) = *self {
            1u8.hash_stable(ctx, hasher);
            value.hash_stable(ctx, hasher);
        } else {
            0u8.hash_stable(ctx, hasher);
        }
    }
}

impl<T1, T2, CTX> HashStable<CTX> for Result<T1, T2>
    where T1: HashStable<CTX>,
          T2: HashStable<CTX>,
{
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(ctx, hasher);
        match *self {
            Ok(ref x) => x.hash_stable(ctx, hasher),
            Err(ref x) => x.hash_stable(ctx, hasher),
        }
    }
}

impl<'a, T, CTX> HashStable<CTX> for &'a T
    where T: HashStable<CTX> + ?Sized
{
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(ctx, hasher);
    }
}

impl<T, CTX> HashStable<CTX> for ::std::mem::Discriminant<T> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        ::std::hash::Hash::hash(self, hasher);
    }
}

impl<I: ::indexed_vec::Idx, T, CTX> HashStable<CTX> for ::indexed_vec::IndexVec<I, T>
    where T: HashStable<CTX>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        self.len().hash_stable(ctx, hasher);
        for v in &self.raw {
            v.hash_stable(ctx, hasher);
        }
    }
}


impl<I: ::indexed_vec::Idx, CTX> HashStable<CTX> for ::indexed_set::IdxSet<I>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          ctx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        self.words().hash_stable(ctx, hasher);
    }
}

impl_stable_hash_via_hash!(::std::path::Path);
impl_stable_hash_via_hash!(::std::path::PathBuf);

impl<K, V, R, HCX> HashStable<HCX> for ::std::collections::HashMap<K, V, R>
    where K: ToStableHashKey<HCX> + Eq + Hash,
          V: HashStable<HCX>,
          R: BuildHasher,
{
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        hash_stable_hashmap(hcx, hasher, self, ToStableHashKey::to_stable_hash_key);
    }
}

impl<K, R, HCX> HashStable<HCX> for ::std::collections::HashSet<K, R>
    where K: ToStableHashKey<HCX> + Eq + Hash,
          R: BuildHasher,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let mut keys: Vec<_> = self.iter()
                                   .map(|k| k.to_stable_hash_key(hcx))
                                   .collect();
        keys.sort_unstable();
        keys.hash_stable(hcx, hasher);
    }
}

impl<K, V, HCX> HashStable<HCX> for ::std::collections::BTreeMap<K, V>
    where K: ToStableHashKey<HCX>,
          V: HashStable<HCX>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let mut entries: Vec<_> = self.iter()
                                      .map(|(k, v)| (k.to_stable_hash_key(hcx), v))
                                      .collect();
        entries.sort_unstable_by(|&(ref sk1, _), &(ref sk2, _)| sk1.cmp(sk2));
        entries.hash_stable(hcx, hasher);
    }
}

impl<K, HCX> HashStable<HCX> for ::std::collections::BTreeSet<K>
    where K: ToStableHashKey<HCX>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let mut keys: Vec<_> = self.iter()
                                   .map(|k| k.to_stable_hash_key(hcx))
                                   .collect();
        keys.sort_unstable();
        keys.hash_stable(hcx, hasher);
    }
}

pub fn hash_stable_hashmap<HCX, K, V, R, SK, F, W>(
    hcx: &mut HCX,
    hasher: &mut StableHasher<W>,
    map: &::std::collections::HashMap<K, V, R>,
    to_stable_hash_key: F)
    where K: Eq + Hash,
          V: HashStable<HCX>,
          R: BuildHasher,
          SK: HashStable<HCX> + Ord + Clone,
          F: Fn(&K, &HCX) -> SK,
          W: StableHasherResult,
{
    let mut entries: Vec<_> = map.iter()
                                  .map(|(k, v)| (to_stable_hash_key(k, hcx), v))
                                  .collect();
    entries.sort_unstable_by(|&(ref sk1, _), &(ref sk2, _)| sk1.cmp(sk2));
    entries.hash_stable(hcx, hasher);
}


/// A vector container that makes sure that its items are hashed in a stable
/// order.
pub struct StableVec<T>(Vec<T>);

impl<T> StableVec<T> {
    pub fn new(v: Vec<T>) -> Self {
        StableVec(v)
    }
}

impl<T> ::std::ops::Deref for StableVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T, HCX> HashStable<HCX> for StableVec<T>
    where T: HashStable<HCX> + ToStableHashKey<HCX>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let StableVec(ref v) = *self;

        let mut sorted: Vec<_> = v.iter()
                                  .map(|x| x.to_stable_hash_key(hcx))
                                  .collect();
        sorted.sort_unstable();
        sorted.hash_stable(hcx, hasher);
    }
}
