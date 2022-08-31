//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

use std::alloc::Allocator;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::path;
use std::rc::Rc;
use std::sync::Arc;

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
pub trait Encoder {
    // Primitive types:
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
    fn emit_i8(&mut self, v: i8);
    fn emit_bool(&mut self, v: bool);
    fn emit_f64(&mut self, v: f64);
    fn emit_f32(&mut self, v: f32);
    fn emit_char(&mut self, v: char);
    fn emit_str(&mut self, v: &str);
    fn emit_raw_bytes(&mut self, s: &[u8]);

    // Convenience for the derive macro:
    fn emit_enum_variant<F>(&mut self, v_id: usize, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.emit_usize(v_id);
        f(self);
    }

    // We put the field index in a const generic to allow the emit_usize to be
    // compiled into a more efficient form. In practice, the variant index is
    // known at compile-time, and that knowledge allows much more efficient
    // codegen than we'd otherwise get. LLVM isn't always able to make the
    // optimization that would otherwise be necessary here, likely due to the
    // multiple levels of inlining and const-prop that are needed.
    #[inline]
    fn emit_fieldless_enum_variant<const ID: usize>(&mut self) {
        self.emit_usize(ID)
    }
}

// Note: all the methods in this trait are infallible, which may be surprising.
// They used to be fallible (i.e. return a `Result`) but many of the impls just
// panicked when something went wrong, and for the cases that didn't the
// top-level invocation would also just panic on failure. Switching to
// infallibility made things faster and lots of code a little simpler and more
// concise.
pub trait Decoder {
    // Primitive types:
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
    fn read_i8(&mut self) -> i8;
    fn read_bool(&mut self) -> bool;
    fn read_f64(&mut self) -> f64;
    fn read_f32(&mut self) -> f32;
    fn read_char(&mut self) -> char;
    fn read_str(&mut self) -> &str;
    fn read_raw_bytes(&mut self, len: usize) -> &[u8];
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
pub trait Encodable<S: Encoder> {
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

    f32 emit_f32 read_f32,
    f64 emit_f64 read_f64,
    bool emit_bool read_bool,
    char emit_char read_char
}

impl<S: Encoder, T: ?Sized> Encodable<S> for &T
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

impl<S: Encoder> Encodable<S> for ::std::num::NonZeroU32 {
    fn encode(&self, s: &mut S) {
        s.emit_u32(self.get());
    }
}

impl<D: Decoder> Decodable<D> for ::std::num::NonZeroU32 {
    fn decode(d: &mut D) -> Self {
        ::std::num::NonZeroU32::new(d.read_u32()).unwrap()
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
    fn decode(_: &mut D) -> () {}
}

impl<S: Encoder, T> Encodable<S> for PhantomData<T> {
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder, T> Decodable<D> for PhantomData<T> {
    fn decode(_: &mut D) -> PhantomData<T> {
        PhantomData
    }
}

impl<D: Decoder, A: Allocator + Default, T: Decodable<D>> Decodable<D> for Box<[T], A> {
    fn decode(d: &mut D) -> Box<[T], A> {
        let v: Vec<T, A> = Decodable::decode(d);
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
        for e in self.iter() {
            e.encode(s);
        }
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Vec<T> {
    fn encode(&self, s: &mut S) {
        let slice: &[T] = self;
        slice.encode(s);
    }
}

impl<D: Decoder, T: Decodable<D>, A: Allocator + Default> Decodable<D> for Vec<T, A> {
    default fn decode(d: &mut D) -> Vec<T, A> {
        let len = d.read_usize();
        let allocator = A::default();
        // SAFETY: we set the capacity in advance, only write elements, and
        // only set the length at the end once the writing has succeeded.
        let mut vec = Vec::with_capacity_in(len, allocator);
        unsafe {
            let ptr: *mut T = vec.as_mut_ptr();
            for i in 0..len {
                std::ptr::write(ptr.add(i), Decodable::decode(d));
            }
            vec.set_len(len);
        }
        vec
    }
}

impl<S: Encoder, T: Encodable<S>, const N: usize> Encodable<S> for [T; N] {
    fn encode(&self, s: &mut S) {
        let slice: &[T] = self;
        slice.encode(s);
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

impl<'a, S: Encoder, T: Encodable<S>> Encodable<S> for Cow<'a, [T]>
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

impl<'a, S: Encoder> Encodable<S> for Cow<'a, str> {
    fn encode(&self, s: &mut S) {
        let val: &str = self;
        val.encode(s)
    }
}

impl<'a, D: Decoder> Decodable<D> for Cow<'a, str> {
    fn decode(d: &mut D) -> Cow<'static, str> {
        let v: String = Decodable::decode(d);
        Cow::Owned(v)
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Option<T> {
    fn encode(&self, s: &mut S) {
        match *self {
            None => s.emit_enum_variant(0, |_| {}),
            Some(ref v) => s.emit_enum_variant(1, |s| v.encode(s)),
        }
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &mut D) -> Option<T> {
        match d.read_usize() {
            0 => None,
            1 => Some(Decodable::decode(d)),
            _ => panic!("Encountered invalid discriminant while decoding `Option`."),
        }
    }
}

impl<S: Encoder, T1: Encodable<S>, T2: Encodable<S>> Encodable<S> for Result<T1, T2> {
    fn encode(&self, s: &mut S) {
        match *self {
            Ok(ref v) => s.emit_enum_variant(0, |s| v.encode(s)),
            Err(ref v) => s.emit_enum_variant(1, |s| v.encode(s)),
        }
    }
}

impl<D: Decoder, T1: Decodable<D>, T2: Decodable<D>> Decodable<D> for Result<T1, T2> {
    fn decode(d: &mut D) -> Result<T1, T2> {
        match d.read_usize() {
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

// FIXME: #15036
// Should use `try_borrow`, returning an
// `encoder.error("attempting to Encode borrowed RefCell")`
// from `encode` when `try_borrow` returns `None`.

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

impl<S: Encoder, T: ?Sized + Encodable<S>, A: Allocator + Default> Encodable<S> for Box<T, A> {
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<D: Decoder, A: Allocator + Default, T: Decodable<D>> Decodable<D> for Box<T, A> {
    fn decode(d: &mut D) -> Box<T, A> {
        let allocator = A::default();
        Box::new_in(Decodable::decode(d), allocator)
    }
}
