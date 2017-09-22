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
use blake2b::Blake2bHasher;
use rustc_serialize::leb128;

fn write_unsigned_leb128_to_buf(buf: &mut [u8; 16], value: u64) -> usize {
    leb128::write_unsigned_leb128_to(value as u128, |i, v| buf[i] = v)
}

fn write_signed_leb128_to_buf(buf: &mut [u8; 16], value: i64) -> usize {
    leb128::write_signed_leb128_to(value as i128, |i, v| buf[i] = v)
}

/// When hashing something that ends up affecting properties like symbol names. We
/// want these symbol names to be calculated independent of other factors like
/// what architecture you're compiling *from*.
///
/// The hashing just uses the standard `Hash` trait, but the implementations of
/// `Hash` for the `usize` and `isize` types are *not* architecture independent
/// (e.g. they has 4 or 8 bytes). As a result we want to avoid `usize` and
/// `isize` completely when hashing.
///
/// To do that, we encode all integers to be hashed with some
/// arch-independent encoding.
///
/// At the moment, we pass i8/u8 straight through and encode
/// all other integers using leb128.
///
/// This hasher currently always uses the stable Blake2b algorithm
/// and allows for variable output lengths through its type
/// parameter.
pub struct StableHasher<W> {
    state: Blake2bHasher,
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
            state: Blake2bHasher::new(mem::size_of::<W>(), &[]),
            bytes_hashed: 0,
            width: PhantomData,
        }
    }

    pub fn finish(self) -> W {
        W::finish(self)
    }
}

impl StableHasherResult for [u8; 20] {
    fn finish(mut hasher: StableHasher<Self>) -> Self {
        let mut result: [u8; 20] = [0; 20];
        result.copy_from_slice(hasher.state.finalize());
        result
    }
}

impl StableHasherResult for u128 {
    fn finish(mut hasher: StableHasher<Self>) -> Self {
        let hash_bytes: &[u8] = hasher.finalize();
        assert!(hash_bytes.len() >= mem::size_of::<u128>());

        unsafe {
            ::std::ptr::read_unaligned(hash_bytes.as_ptr() as *const u128)
        }
    }
}

impl StableHasherResult for u64 {
    fn finish(mut hasher: StableHasher<Self>) -> Self {
        hasher.state.finalize();
        hasher.state.finish()
    }
}

impl<W> StableHasher<W> {
    #[inline]
    pub fn finalize(&mut self) -> &[u8] {
        self.state.finalize()
    }

    #[inline]
    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
    }

    #[inline]
    fn write_uleb128(&mut self, value: u64) {
        let mut buf = [0; 16];
        let len = write_unsigned_leb128_to_buf(&mut buf, value);
        self.state.write(&buf[..len]);
        self.bytes_hashed += len as u64;
    }

    #[inline]
    fn write_ileb128(&mut self, value: i64) {
        let mut buf = [0; 16];
        let len = write_signed_leb128_to_buf(&mut buf, value);
        self.state.write(&buf[..len]);
        self.bytes_hashed += len as u64;
    }
}

// For the non-u8 integer cases we leb128 encode them first. Because small
// integers dominate, this significantly and cheaply reduces the number of
// bytes hashed, which is good because blake2b is expensive.
impl<W> Hasher for StableHasher<W> {
    fn finish(&self) -> u64 {
        panic!("use StableHasher::finish instead");
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
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write_uleb128(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_uleb128(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
        self.bytes_hashed += 1;
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.write_ileb128(i as i64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write_ileb128(i as i64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_ileb128(i);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.write_ileb128(i as i64);
    }
}


/// Something that can provide a stable hashing context.
pub trait StableHashingContextProvider {
    type ContextType;
    fn create_stable_hashing_context(&self) -> Self::ContextType;
}

impl<'a, T: StableHashingContextProvider> StableHashingContextProvider for &'a T {
    type ContextType = T::ContextType;

    fn create_stable_hashing_context(&self) -> Self::ContextType {
        (**self).create_stable_hashing_context()
    }
}

impl<'a, T: StableHashingContextProvider> StableHashingContextProvider for &'a mut T {
    type ContextType = T::ContextType;

    fn create_stable_hashing_context(&self) -> Self::ContextType {
        (**self).create_stable_hashing_context()
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
macro_rules! impl_stable_hash_via_hash {
    ($t:ty) => (
        impl<CTX> HashStable<CTX> for $t {
            #[inline]
            fn hash_stable<W: StableHasherResult>(&self,
                                                  _: &mut CTX,
                                                  hasher: &mut StableHasher<W>) {
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


impl<I: ::indexed_vec::Idx, CTX> HashStable<CTX> for ::indexed_set::IdxSetBuf<I>
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

