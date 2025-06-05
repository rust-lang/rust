//! Support code for encoding and decoding types.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::num::NonZero;
use std::path;
use std::rc::Rc;
use std::sync::Arc;

use rustc_hashes::{Hash64, Hash128};
use smallvec::{Array, SmallVec};
use thin_vec::ThinVec;

/// A byte that [cannot occur in UTF8 sequences][utf8]. Used to mark the end of a string.
/// This way we can skip validation and still be relatively sure that deserialization
/// did not desynchronize.
///
/// [utf8]: https://en.wikipedia.org/w/index.php?title=UTF-8&oldid=1058865525#Codepage_layout
const STR_SENTINEL: u8 = 0xC1;

/// A note about error handling.
///
/// Encoders may be fallible, but in practice failure is rare and there are so
/// many nested calls that typical Rust error handling (via `Result` and `?`)
/// is pervasive and has non-trivial cost. Instead, impls of this trait must
/// implement a delayed error handling strategy. If a failure occurs, they
/// should record this internally, and all subsequent encoding operations can
/// be processed or ignored, whichever is appropriate. Then they should provide
/// a `finish` method that finishes up encoding. If the encoder is fallible,
/// `finish` should return a `Result` that indicates success or failure.
///
/// This current does not support `f32` nor `f64`, as they're not needed in any
/// serialized data structures. That could be changed, but consider whether it
/// really makes sense to store floating-point values at all.
/// (If you need it, revert <https://github.com/rust-lang/rust/pull/109984>.)
pub trait Encoder {
    fn emit_usize(&mut self, v: usize);
    fn emit_u128(&mut self, v: u128);
    fn emit_u64(&mut self, v: u64);
    fn emit_u32(&mut self, v: u32);
    fn emit_u16(&mut self, v: u16);
    fn emit_u8(&mut self, v: u8);

    fn emit_isize(&mut self, v: isize);
    fn emit_i128(&mut self, v: i128);
    fn emit_i64(&mut self, v: i64);
    fn emit_i32(&mut self, v: i32);
    fn emit_i16(&mut self, v: i16);

    #[inline]
    fn emit_i8(&mut self, v: i8) {
        self.emit_u8(v as u8);
    }

    #[inline]
    fn emit_bool(&mut self, v: bool) {
        self.emit_u8(if v { 1 } else { 0 });
    }

    #[inline]
    fn emit_char(&mut self, v: char) {
        self.emit_u32(v as u32);
    }

    #[inline]
    fn emit_str(&mut self, v: &str) {
        self.emit_usize(v.len());
        self.emit_raw_bytes(v.as_bytes());
        self.emit_u8(STR_SENTINEL);
    }

    fn emit_raw_bytes(&mut self, s: &[u8]);
}

// Note: all the methods in this trait are infallible, which may be surprising.
// They used to be fallible (i.e. return a `Result`) but many of the impls just
// panicked when something went wrong, and for the cases that didn't the
// top-level invocation would also just panic on failure. Switching to
// infallibility made things faster and lots of code a little simpler and more
// concise.
///
/// This current does not support `f32` nor `f64`, as they're not needed in any
/// serialized data structures. That could be changed, but consider whether it
/// really makes sense to store floating-point values at all.
/// (If you need it, revert <https://github.com/rust-lang/rust/pull/109984>.)
pub trait Decoder {
    fn read_usize(&mut self) -> usize;
    fn read_u128(&mut self) -> u128;
    fn read_u64(&mut self) -> u64;
    fn read_u32(&mut self) -> u32;
    fn read_u16(&mut self) -> u16;
    fn read_u8(&mut self) -> u8;

    fn read_isize(&mut self) -> isize;
    fn read_i128(&mut self) -> i128;
    fn read_i64(&mut self) -> i64;
    fn read_i32(&mut self) -> i32;
    fn read_i16(&mut self) -> i16;

    #[inline]
    fn read_i8(&mut self) -> i8 {
        self.read_u8() as i8
    }

    #[inline]
    fn read_bool(&mut self) -> bool {
        let value = self.read_u8();
        value != 0
    }

    #[inline]
    fn read_char(&mut self) -> char {
        let bits = self.read_u32();
        std::char::from_u32(bits).unwrap()
    }

    #[inline]
    fn read_str(&mut self) -> &str {
        let len = self.read_usize();
        let bytes = self.read_raw_bytes(len + 1);
        assert!(bytes[len] == STR_SENTINEL);
        unsafe { std::str::from_utf8_unchecked(&bytes[..len]) }
    }

    fn read_raw_bytes(&mut self, len: usize) -> &[u8];

    fn peek_byte(&self) -> u8;
    fn position(&self) -> usize;
}

/// Trait for types that can be serialized
///
/// This can be implemented using the `Encodable`, `TyEncodable` and
/// `MetadataEncodable` macros.
///
/// * `Encodable` should be used in crates that don't depend on
///   `rustc_middle`.
/// * `MetadataEncodable` is used in `rustc_metadata` for types that contain
///   `rustc_metadata::rmeta::Lazy`.
/// * `TyEncodable` should be used for types that are only serialized in crate
///   metadata or the incremental cache. This is most types in `rustc_middle`.
pub trait Encodable<S: Encoder>: crate::PointeeSized {
    fn encode(&self, s: &mut S);
}

/// Trait for types that can be deserialized
///
/// This can be implemented using the `Decodable`, `TyDecodable` and
/// `MetadataDecodable` macros.
///
/// * `Decodable` should be used in crates that don't depend on
///   `rustc_middle`.
/// * `MetadataDecodable` is used in `rustc_metadata` for types that contain
///   `rustc_metadata::rmeta::Lazy`.
/// * `TyDecodable` should be used for types that are only serialized in crate
///   metadata or the incremental cache. This is most types in `rustc_middle`.
pub trait Decodable<D: Decoder>: Sized {
    fn decode(d: &mut D) -> Self;
}

macro_rules! direct_serialize_impls {
    ($($ty:ident $emit_method:ident $read_method:ident),*) => {
        $(
            impl<S: Encoder> Encodable<S> for $ty {
                fn encode(&self, s: &mut S) {
                    s.$emit_method(*self);
                }
            }

            impl<D: Decoder> Decodable<D> for $ty {
                fn decode(d: &mut D) -> $ty {
                    d.$read_method()
                }
            }
        )*
    }
}

direct_serialize_impls! {
    usize emit_usize read_usize,
    u8 emit_u8 read_u8,
    u16 emit_u16 read_u16,
    u32 emit_u32 read_u32,
    u64 emit_u64 read_u64,
    u128 emit_u128 read_u128,

    isize emit_isize read_isize,
    i8 emit_i8 read_i8,
    i16 emit_i16 read_i16,
    i32 emit_i32 read_i32,
    i64 emit_i64 read_i64,
    i128 emit_i128 read_i128,

    bool emit_bool read_bool,
    char emit_char read_char
}

impl<S: Encoder, T: ?Sized + crate::PointeeSized> Encodable<S> for &T
where
    T: Encodable<S>,
{
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<S: Encoder> Encodable<S> for ! {
    fn encode(&self, _s: &mut S) {
        unreachable!();
    }
}

impl<D: Decoder> Decodable<D> for ! {
    fn decode(_d: &mut D) -> ! {
        unreachable!()
    }
}

impl<S: Encoder> Encodable<S> for NonZero<u32> {
    fn encode(&self, s: &mut S) {
        s.emit_u32(self.get());
    }
}

impl<D: Decoder> Decodable<D> for NonZero<u32> {
    fn decode(d: &mut D) -> Self {
        NonZero::new(d.read_u32()).unwrap()
    }
}

impl<S: Encoder> Encodable<S> for str {
    fn encode(&self, s: &mut S) {
        s.emit_str(self);
    }
}

impl<S: Encoder> Encodable<S> for String {
    fn encode(&self, s: &mut S) {
        s.emit_str(&self[..]);
    }
}

impl<D: Decoder> Decodable<D> for String {
    fn decode(d: &mut D) -> String {
        d.read_str().to_owned()
    }
}

impl<S: Encoder> Encodable<S> for () {
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for () {
    fn decode(_: &mut D) {}
}

impl<S: Encoder, T> Encodable<S> for PhantomData<T> {
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder, T> Decodable<D> for PhantomData<T> {
    fn decode(_: &mut D) -> PhantomData<T> {
        PhantomData
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Box<[T]> {
    fn decode(d: &mut D) -> Box<[T]> {
        let v: Vec<T> = Decodable::decode(d);
        v.into_boxed_slice()
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Rc<T> {
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Rc<T> {
    fn decode(d: &mut D) -> Rc<T> {
        Rc::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for [T] {
    default fn encode(&self, s: &mut S) {
        s.emit_usize(self.len());
        for e in self {
            e.encode(s);
        }
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Vec<T> {
    fn encode(&self, s: &mut S) {
        self.as_slice().encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Vec<T> {
    default fn decode(d: &mut D) -> Vec<T> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<S: Encoder, T: Encodable<S>, const N: usize> Encodable<S> for [T; N] {
    fn encode(&self, s: &mut S) {
        self.as_slice().encode(s);
    }
}

impl<D: Decoder, const N: usize> Decodable<D> for [u8; N] {
    fn decode(d: &mut D) -> [u8; N] {
        let len = d.read_usize();
        assert!(len == N);
        let mut v = [0u8; N];
        for i in 0..len {
            v[i] = Decodable::decode(d);
        }
        v
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Cow<'_, [T]>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn encode(&self, s: &mut S) {
        let slice: &[T] = self;
        slice.encode(s);
    }
}

impl<D: Decoder, T: Decodable<D> + ToOwned> Decodable<D> for Cow<'static, [T]>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn decode(d: &mut D) -> Cow<'static, [T]> {
        let v: Vec<T> = Decodable::decode(d);
        Cow::Owned(v)
    }
}

impl<S: Encoder> Encodable<S> for Cow<'_, str> {
    fn encode(&self, s: &mut S) {
        let val: &str = self;
        val.encode(s)
    }
}

impl<D: Decoder> Decodable<D> for Cow<'_, str> {
    fn decode(d: &mut D) -> Cow<'static, str> {
        let v: String = Decodable::decode(d);
        Cow::Owned(v)
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Option<T> {
    fn encode(&self, s: &mut S) {
        match *self {
            None => s.emit_u8(0),
            Some(ref v) => {
                s.emit_u8(1);
                v.encode(s);
            }
        }
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &mut D) -> Option<T> {
        match d.read_u8() {
            0 => None,
            1 => Some(Decodable::decode(d)),
            _ => panic!("Encountered invalid discriminant while decoding `Option`."),
        }
    }
}

impl<S: Encoder, T1: Encodable<S>, T2: Encodable<S>> Encodable<S> for Result<T1, T2> {
    fn encode(&self, s: &mut S) {
        match *self {
            Ok(ref v) => {
                s.emit_u8(0);
                v.encode(s);
            }
            Err(ref v) => {
                s.emit_u8(1);
                v.encode(s);
            }
        }
    }
}

impl<D: Decoder, T1: Decodable<D>, T2: Decodable<D>> Decodable<D> for Result<T1, T2> {
    fn decode(d: &mut D) -> Result<T1, T2> {
        match d.read_u8() {
            0 => Ok(T1::decode(d)),
            1 => Err(T2::decode(d)),
            _ => panic!("Encountered invalid discriminant while decoding `Result`."),
        }
    }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        impl<D: Decoder, $($name: Decodable<D>),+> Decodable<D> for ($($name,)+) {
            fn decode(d: &mut D) -> ($($name,)+) {
                ($({ let element: $name = Decodable::decode(d); element },)+)
            }
        }
        impl<S: Encoder, $($name: Encodable<S>),+> Encodable<S> for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode(&self, s: &mut S) {
                let ($(ref $name,)+) = *self;
                $($name.encode(s);)+
            }
        }
        peel! { $($name,)+ }
    )
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

impl<S: Encoder> Encodable<S> for path::Path {
    fn encode(&self, e: &mut S) {
        self.to_str().unwrap().encode(e);
    }
}

impl<S: Encoder> Encodable<S> for path::PathBuf {
    fn encode(&self, e: &mut S) {
        path::Path::encode(self, e);
    }
}

impl<D: Decoder> Decodable<D> for path::PathBuf {
    fn decode(d: &mut D) -> path::PathBuf {
        let bytes: String = Decodable::decode(d);
        path::PathBuf::from(bytes)
    }
}

impl<S: Encoder, T: Encodable<S> + Copy> Encodable<S> for Cell<T> {
    fn encode(&self, s: &mut S) {
        self.get().encode(s);
    }
}

impl<D: Decoder, T: Decodable<D> + Copy> Decodable<D> for Cell<T> {
    fn decode(d: &mut D) -> Cell<T> {
        Cell::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for RefCell<T> {
    fn encode(&self, s: &mut S) {
        self.borrow().encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for RefCell<T> {
    fn decode(d: &mut D) -> RefCell<T> {
        RefCell::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Arc<T> {
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Arc<T> {
    fn decode(d: &mut D) -> Arc<T> {
        Arc::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: ?Sized + Encodable<S>> Encodable<S> for Box<T> {
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Box<T> {
    fn decode(d: &mut D) -> Box<T> {
        Box::new(Decodable::decode(d))
    }
}

impl<S: Encoder, A: Array<Item: Encodable<S>>> Encodable<S> for SmallVec<A> {
    fn encode(&self, s: &mut S) {
        self.as_slice().encode(s);
    }
}

impl<D: Decoder, A: Array<Item: Decodable<D>>> Decodable<D> for SmallVec<A> {
    fn decode(d: &mut D) -> SmallVec<A> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for ThinVec<T> {
    fn encode(&self, s: &mut S) {
        self.as_slice().encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for ThinVec<T> {
    fn decode(d: &mut D) -> ThinVec<T> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for VecDeque<T> {
    fn encode(&self, s: &mut S) {
        s.emit_usize(self.len());
        for e in self {
            e.encode(s);
        }
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for VecDeque<T> {
    fn decode(d: &mut D) -> VecDeque<T> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<S: Encoder, K, V> Encodable<S> for BTreeMap<K, V>
where
    K: Encodable<S> + PartialEq + Ord,
    V: Encodable<S>,
{
    fn encode(&self, e: &mut S) {
        e.emit_usize(self.len());
        for (key, val) in self {
            key.encode(e);
            val.encode(e);
        }
    }
}

impl<D: Decoder, K, V> Decodable<D> for BTreeMap<K, V>
where
    K: Decodable<D> + PartialEq + Ord,
    V: Decodable<D>,
{
    fn decode(d: &mut D) -> BTreeMap<K, V> {
        let len = d.read_usize();
        (0..len).map(|_| (Decodable::decode(d), Decodable::decode(d))).collect()
    }
}

impl<S: Encoder, T> Encodable<S> for BTreeSet<T>
where
    T: Encodable<S> + PartialEq + Ord,
{
    fn encode(&self, s: &mut S) {
        s.emit_usize(self.len());
        for e in self {
            e.encode(s);
        }
    }
}

impl<D: Decoder, T> Decodable<D> for BTreeSet<T>
where
    T: Decodable<D> + PartialEq + Ord,
{
    fn decode(d: &mut D) -> BTreeSet<T> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<E: Encoder, K, V, S> Encodable<E> for HashMap<K, V, S>
where
    K: Encodable<E> + Eq,
    V: Encodable<E>,
    S: BuildHasher,
{
    fn encode(&self, e: &mut E) {
        e.emit_usize(self.len());
        for (key, val) in self {
            key.encode(e);
            val.encode(e);
        }
    }
}

impl<D: Decoder, K, V, S> Decodable<D> for HashMap<K, V, S>
where
    K: Decodable<D> + Hash + Eq,
    V: Decodable<D>,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> HashMap<K, V, S> {
        let len = d.read_usize();
        (0..len).map(|_| (Decodable::decode(d), Decodable::decode(d))).collect()
    }
}

impl<E: Encoder, T, S> Encodable<E> for HashSet<T, S>
where
    T: Encodable<E> + Eq,
    S: BuildHasher,
{
    fn encode(&self, s: &mut E) {
        s.emit_usize(self.len());
        for e in self {
            e.encode(s);
        }
    }
}

impl<D: Decoder, T, S> Decodable<D> for HashSet<T, S>
where
    T: Decodable<D> + Hash + Eq,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> HashSet<T, S> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<E: Encoder, K, V, S> Encodable<E> for indexmap::IndexMap<K, V, S>
where
    K: Encodable<E> + Hash + Eq,
    V: Encodable<E>,
    S: BuildHasher,
{
    fn encode(&self, e: &mut E) {
        e.emit_usize(self.len());
        for (key, val) in self {
            key.encode(e);
            val.encode(e);
        }
    }
}

impl<D: Decoder, K, V, S> Decodable<D> for indexmap::IndexMap<K, V, S>
where
    K: Decodable<D> + Hash + Eq,
    V: Decodable<D>,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> indexmap::IndexMap<K, V, S> {
        let len = d.read_usize();
        (0..len).map(|_| (Decodable::decode(d), Decodable::decode(d))).collect()
    }
}

impl<E: Encoder, T, S> Encodable<E> for indexmap::IndexSet<T, S>
where
    T: Encodable<E> + Hash + Eq,
    S: BuildHasher,
{
    fn encode(&self, s: &mut E) {
        s.emit_usize(self.len());
        for e in self {
            e.encode(s);
        }
    }
}

impl<D: Decoder, T, S> Decodable<D> for indexmap::IndexSet<T, S>
where
    T: Decodable<D> + Hash + Eq,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> indexmap::IndexSet<T, S> {
        let len = d.read_usize();
        (0..len).map(|_| Decodable::decode(d)).collect()
    }
}

impl<E: Encoder, T: Encodable<E>> Encodable<E> for Rc<[T]> {
    fn encode(&self, s: &mut E) {
        let slice: &[T] = self;
        slice.encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Rc<[T]> {
    fn decode(d: &mut D) -> Rc<[T]> {
        let vec: Vec<T> = Decodable::decode(d);
        vec.into()
    }
}

impl<E: Encoder, T: Encodable<E>> Encodable<E> for Arc<[T]> {
    fn encode(&self, s: &mut E) {
        let slice: &[T] = self;
        slice.encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Arc<[T]> {
    fn decode(d: &mut D) -> Arc<[T]> {
        let vec: Vec<T> = Decodable::decode(d);
        vec.into()
    }
}

impl<S: Encoder> Encodable<S> for Hash64 {
    #[inline]
    fn encode(&self, s: &mut S) {
        s.emit_raw_bytes(&self.as_u64().to_le_bytes());
    }
}

impl<S: Encoder> Encodable<S> for Hash128 {
    #[inline]
    fn encode(&self, s: &mut S) {
        s.emit_raw_bytes(&self.as_u128().to_le_bytes());
    }
}

impl<D: Decoder> Decodable<D> for Hash64 {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self::new(u64::from_le_bytes(d.read_raw_bytes(8).try_into().unwrap()))
    }
}

impl<D: Decoder> Decodable<D> for Hash128 {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self::new(u128::from_le_bytes(d.read_raw_bytes(16).try_into().unwrap()))
    }
}
