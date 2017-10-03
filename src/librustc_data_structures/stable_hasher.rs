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
use std::num::Wrapping;
use std::mem;
use std::ptr;

use siphasher::sip128::{SipHasher, Hasher128};

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
    state: SipHasher,
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
            state: SipHasher::new(),
            bytes_hashed: 0,
            width: PhantomData,
        }
    }

    pub fn finish(self) -> W {
        W::finish(self)
    }
}

impl StableHasherResult for [u8; 20] {
    fn finish(hasher: StableHasher<Self>) -> Self {
        let hash = hasher.state.finish128();

        [
            (hash.h1 <<  0) as u8,
            (hash.h1 <<  8) as u8,
            (hash.h1 << 16) as u8,
            (hash.h1 << 24) as u8,
            (hash.h1 << 32) as u8,
            (hash.h1 << 40) as u8,
            (hash.h1 << 48) as u8,
            (hash.h1 << 56) as u8,

            13,
            29,
            119,
            231,

            (hash.h2 <<  0) as u8,
            (hash.h2 <<  8) as u8,
            (hash.h2 << 16) as u8,
            (hash.h2 << 24) as u8,
            (hash.h2 << 32) as u8,
            (hash.h2 << 40) as u8,
            (hash.h2 << 48) as u8,
            (hash.h2 << 56) as u8,
        ]
    }
}

impl StableHasherResult for u128 {
    #[inline]
    fn finish(hasher: StableHasher<Self>) -> Self {
        let h = hasher.state.finish128();
        (h.h1 as u128) | ((h.h2 as u128) << 64)
    }
}

impl StableHasherResult for u64 {
    #[inline]
    fn finish(hasher: StableHasher<Self>) -> Self {
        hasher.state.finish128().h1
    }
}

impl StableHasherResult for (u64, u64) {
    #[inline]
    fn finish(hasher: StableHasher<Self>) -> Self {
        let h = hasher.state.finish128();
        (h.h1, h.h2)
    }
}

impl<W> StableHasher<W> {
    #[inline]
    pub fn finalize(&mut self) -> (u64, u64) {
        let h = self.state.finish128();
        (h.h1, h.h2)
    }

    #[inline]
    pub fn bytes_hashed(&self) -> u64 {
        self.bytes_hashed
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
        self.state.write_usize(i.to_le());
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
        self.state.write_isize(i.to_le());
        self.bytes_hashed += 8;
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






macro_rules! impl_read {
    ($fn_name: ident, $ty: ty) => (
        #[inline(always)]
        pub fn $fn_name(ptr_addr: usize) -> Wrapping<u64> {
            let ptr: *const $ty = ptr_addr as *const $ty;
            Wrapping(unsafe { *ptr as u64 })
        }
    )
}

impl_read!(read_u64, u64);
impl_read!(read_u32, u32);
impl_read!(read_u16, u16);
impl_read!(read_u8, u8);

#[inline(always)]
pub fn rotate_right(v: Wrapping<u64>, k: u32) -> Wrapping<u64> {
    Wrapping(v.0.rotate_right(k))
}


const K0: Wrapping<u64> = Wrapping(0xC83A91E1);
const K1: Wrapping<u64> = Wrapping(0x8648DBDB);
const K2: Wrapping<u64> = Wrapping(0x7BDEC03B);
const K3: Wrapping<u64> = Wrapping(0x2F5870A5);

pub struct MetroHash128 {
    v: [Wrapping<u64>; 4],
    b: [u64; 4],
    bytes: usize,
}

impl Default for MetroHash128 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MetroHash128 {
    #[inline]
    pub fn new() -> MetroHash128 {
        Self::with_seed(0)
    }

    #[inline]
    pub fn with_seed(seed: u64) -> MetroHash128 {
        let seed = Wrapping(seed);
        MetroHash128 {
            b: unsafe { mem::uninitialized() },
            v: [(seed - K0) * K3,
                (seed + K1) * K2,
                (seed + K0) * K2,
                (seed - K1) * K3],
            bytes: 0,
        }
    }

    #[inline]
    fn finish128(&self) -> (u64, u64) {
        // copy internal state
        let mut v = self.v;

        // finalize bulk loop, if used
        if self.bytes >= 32 {
            v[2] = v[2] ^ (rotate_right(((v[0] + v[3]) * K0) + v[1], 21) * K1);
            v[3] = v[3] ^ (rotate_right(((v[1] + v[2]) * K1) + v[0], 21) * K0);
            v[0] = v[0] ^ (rotate_right(((v[0] + v[2]) * K0) + v[3], 21) * K1);
            v[1] = v[1] ^ (rotate_right(((v[1] + v[3]) * K1) + v[2], 21) * K0);
        }

        // process any self.bytes remaining in the input buffer
        let mut ptr = &self.b as *const _ as usize;
        let end = ptr + self.bytes % 32;

        if (end - ptr) >= 16 {
            v[0] = v[0] + (read_u64(ptr) * K2);
            ptr += 8;
            v[0] = rotate_right(v[0], 33) * K3;
            v[1] = v[1] + (read_u64(ptr) * K2);
            ptr += 8;
            v[1] = rotate_right(v[1], 33) * K3;
            v[0] = v[0] ^ (rotate_right((v[0] * K2) + v[1], 45) * K1);
            v[1] = v[1] ^ (rotate_right((v[1] * K3) + v[0], 45) * K0);
        }

        if (end - ptr) >= 8 {
            v[0] = v[0] + (read_u64(ptr) * K2);
            ptr += 8;
            v[0] = rotate_right(v[0], 33) * K3;
            v[0] = v[0] ^ (rotate_right((v[0] * K2) + v[1], 27) * K1);
        }

        if (end - ptr) >= 4 {
            v[1] = v[1] + (read_u32(ptr) * K2);
            ptr += 4;
            v[1] = rotate_right(v[1], 33) * K3;
            v[1] = v[1] ^ (rotate_right((v[1] * K3) + v[0], 46) * K0);
        }

        if (end - ptr) >= 2 {
            v[0] = v[0] + (read_u16(ptr) * K2);
            ptr += 2;
            v[0] = rotate_right(v[0], 33) * K3;
            v[0] = v[0] ^ (rotate_right((v[0] * K2) + v[1], 22) * K1);
        }

        if (end - ptr) >= 1 {
            v[1] = v[1] + (read_u8(ptr) * K2);
            v[1] = rotate_right(v[1], 33) * K3;
            v[1] = v[1] ^ (rotate_right((v[1] * K3) + v[0], 58) * K0);
        }

        v[0] = v[0] + (rotate_right((v[0] * K0) + v[1], 13));
        v[1] = v[1] + (rotate_right((v[1] * K1) + v[0], 37));
        v[0] = v[0] + (rotate_right((v[0] * K2) + v[1], 13));
        v[1] = v[1] + (rotate_right((v[1] * K3) + v[0], 37));

        (v[0].0, v[1].0)
    }
}

impl Hasher for MetroHash128 {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut ptr = bytes.as_ptr() as usize;
        let end = ptr + bytes.len();
        // input buffer may be partially filled
        if self.bytes % 32 != 0 {
            let mut fill = 32 - (self.bytes % 32);
            if fill > bytes.len() {
                fill = bytes.len();
            }

            unsafe {
                ptr::copy_nonoverlapping(ptr as *const u8,
                                         (&mut self.b[0] as *mut _ as *mut u8).offset((self.bytes %
                                                                                       32) as
                                                                                      isize),
                                         fill);
            }
            ptr += fill;
            self.bytes += fill;

            // input buffer is still partially filled
            if self.bytes % 32 != 0 {
                return;
            }

            // process full input buffer
            self.v[0] = self.v[0] + (read_u64(&self.b[0] as *const _ as usize) * K0);
            self.v[0] = rotate_right(self.v[0], 29) + self.v[2];
            self.v[1] = self.v[1] + (read_u64(&self.b[1] as *const _ as usize) * K1);
            self.v[1] = rotate_right(self.v[1], 29) + self.v[3];
            self.v[2] = self.v[2] + (read_u64(&self.b[2] as *const _ as usize) * K2);
            self.v[2] = rotate_right(self.v[2], 29) + self.v[0];
            self.v[3] = self.v[3] + (read_u64(&self.b[3] as *const _ as usize) * K3);
            self.v[3] = rotate_right(self.v[3], 29) + self.v[1];
        }

        // bulk update
        self.bytes += end - ptr;
        while ptr + 32 <= end {
            // process directly from the source, bypassing the input buffer
            self.v[0] = self.v[0] + (read_u64(ptr) * K0);
            ptr += 8;
            self.v[0] = rotate_right(self.v[0], 29) + self.v[2];
            self.v[1] = self.v[1] + (read_u64(ptr) * K1);
            ptr += 8;
            self.v[1] = rotate_right(self.v[1], 29) + self.v[3];
            self.v[2] = self.v[2] + (read_u64(ptr) * K2);
            ptr += 8;
            self.v[2] = rotate_right(self.v[2], 29) + self.v[0];
            self.v[3] = self.v[3] + (read_u64(ptr) * K3);
            ptr += 8;
            self.v[3] = rotate_right(self.v[3], 29) + self.v[1];
        }

        // store remaining self.bytes in input buffer
        if ptr < end {
            unsafe {
                ptr::copy_nonoverlapping(ptr as *const u8,
                                         &mut self.b[0] as *mut _ as *mut u8,
                                         end - ptr);
            }
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.finish128().0
    }
}
