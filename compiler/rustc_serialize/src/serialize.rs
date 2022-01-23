//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::path;
use std::rc::Rc;
use std::sync::Arc;

pub trait Encoder {
    type Error;

    // Primitive types:
    fn emit_unit(&mut self) -> Result<(), Self::Error>;
    fn emit_usize(&mut self, v: usize) -> Result<(), Self::Error>;
    fn emit_u128(&mut self, v: u128) -> Result<(), Self::Error>;
    fn emit_u64(&mut self, v: u64) -> Result<(), Self::Error>;
    fn emit_u32(&mut self, v: u32) -> Result<(), Self::Error>;
    fn emit_u16(&mut self, v: u16) -> Result<(), Self::Error>;
    fn emit_u8(&mut self, v: u8) -> Result<(), Self::Error>;
    fn emit_isize(&mut self, v: isize) -> Result<(), Self::Error>;
    fn emit_i128(&mut self, v: i128) -> Result<(), Self::Error>;
    fn emit_i64(&mut self, v: i64) -> Result<(), Self::Error>;
    fn emit_i32(&mut self, v: i32) -> Result<(), Self::Error>;
    fn emit_i16(&mut self, v: i16) -> Result<(), Self::Error>;
    fn emit_i8(&mut self, v: i8) -> Result<(), Self::Error>;
    fn emit_bool(&mut self, v: bool) -> Result<(), Self::Error>;
    fn emit_f64(&mut self, v: f64) -> Result<(), Self::Error>;
    fn emit_f32(&mut self, v: f32) -> Result<(), Self::Error>;
    fn emit_char(&mut self, v: char) -> Result<(), Self::Error>;
    fn emit_str(&mut self, v: &str) -> Result<(), Self::Error>;
    fn emit_raw_bytes(&mut self, s: &[u8]) -> Result<(), Self::Error>;

    // Compound types:
    #[inline]
    fn emit_enum<F>(&mut self, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    fn emit_enum_variant<F>(
        &mut self,
        _v_name: &str,
        v_id: usize,
        _len: usize,
        f: F,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        self.emit_usize(v_id)?;
        f(self)
    }

    // We put the field index in a const generic to allow the emit_usize to be
    // compiled into a more efficient form. In practice, the variant index is
    // known at compile-time, and that knowledge allows much more efficient
    // codegen than we'd otherwise get. LLVM isn't always able to make the
    // optimization that would otherwise be necessary here, likely due to the
    // multiple levels of inlining and const-prop that are needed.
    #[inline]
    fn emit_fieldless_enum_variant<const ID: usize>(
        &mut self,
        _v_name: &str,
    ) -> Result<(), Self::Error> {
        self.emit_usize(ID)
    }

    #[inline]
    fn emit_enum_variant_arg<F>(&mut self, _first: bool, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    #[inline]
    fn emit_struct<F>(&mut self, _no_fields: bool, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    #[inline]
    fn emit_struct_field<F>(&mut self, _f_name: &str, _first: bool, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    #[inline]
    fn emit_tuple<F>(&mut self, _len: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    #[inline]
    fn emit_tuple_arg<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    // Specialized types:
    fn emit_option<F>(&mut self, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        self.emit_enum(f)
    }

    #[inline]
    fn emit_option_none(&mut self) -> Result<(), Self::Error> {
        self.emit_enum_variant("None", 0, 0, |_| Ok(()))
    }

    fn emit_option_some<F>(&mut self, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        self.emit_enum_variant("Some", 1, 1, f)
    }

    fn emit_seq<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        self.emit_usize(len)?;
        f(self)
    }

    #[inline]
    fn emit_seq_elt<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    fn emit_map<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        self.emit_usize(len)?;
        f(self)
    }

    #[inline]
    fn emit_map_elt_key<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
    }

    #[inline]
    fn emit_map_elt_val<F>(&mut self, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Self::Error>,
    {
        f(self)
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
    fn read_unit(&mut self) -> ();
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
    fn read_str(&mut self) -> Cow<'_, str>;
    fn read_raw_bytes_into(&mut self, s: &mut [u8]);

    // Compound types:
    #[inline]
    fn read_enum<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_enum_variant<T, F>(&mut self, _names: &[&str], mut f: F) -> T
    where
        F: FnMut(&mut Self, usize) -> T,
    {
        let disr = self.read_usize();
        f(self, disr)
    }

    #[inline]
    fn read_enum_variant_arg<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_struct<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_struct_field<T, F>(&mut self, _f_name: &str, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_tuple<T, F>(&mut self, _len: usize, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_tuple_arg<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    // Specialized types:
    fn read_option<T, F>(&mut self, mut f: F) -> T
    where
        F: FnMut(&mut Self, bool) -> T,
    {
        self.read_enum(move |this| {
            this.read_enum_variant(&["None", "Some"], move |this, idx| match idx {
                0 => f(this, false),
                1 => f(this, true),
                _ => panic!("read_option: expected 0 for None or 1 for Some"),
            })
        })
    }

    fn read_seq<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self, usize) -> T,
    {
        let len = self.read_usize();
        f(self, len)
    }

    #[inline]
    fn read_seq_elt<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    fn read_map<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self, usize) -> T,
    {
        let len = self.read_usize();
        f(self, len)
    }

    #[inline]
    fn read_map_elt_key<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }

    #[inline]
    fn read_map_elt_val<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        f(self)
    }
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
    fn encode(&self, s: &mut S) -> Result<(), S::Error>;
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
                fn encode(&self, s: &mut S) -> Result<(), S::Error> {
                    s.$emit_method(*self)
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

impl<S: Encoder> Encodable<S> for ! {
    fn encode(&self, _s: &mut S) -> Result<(), S::Error> {
        unreachable!()
    }
}

impl<D: Decoder> Decodable<D> for ! {
    fn decode(_d: &mut D) -> ! {
        unreachable!()
    }
}

impl<S: Encoder> Encodable<S> for ::std::num::NonZeroU32 {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.get())
    }
}

impl<D: Decoder> Decodable<D> for ::std::num::NonZeroU32 {
    fn decode(d: &mut D) -> Self {
        ::std::num::NonZeroU32::new(d.read_u32()).unwrap()
    }
}

impl<S: Encoder> Encodable<S> for str {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self)
    }
}

impl<S: Encoder> Encodable<S> for &str {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self)
    }
}

impl<S: Encoder> Encodable<S> for String {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self[..])
    }
}

impl<D: Decoder> Decodable<D> for String {
    fn decode(d: &mut D) -> String {
        d.read_str().into_owned()
    }
}

impl<S: Encoder> Encodable<S> for () {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_unit()
    }
}

impl<D: Decoder> Decodable<D> for () {
    fn decode(d: &mut D) -> () {
        d.read_unit()
    }
}

impl<S: Encoder, T> Encodable<S> for PhantomData<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_unit()
    }
}

impl<D: Decoder, T> Decodable<D> for PhantomData<T> {
    fn decode(d: &mut D) -> PhantomData<T> {
        d.read_unit();
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
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Rc<T> {
    fn decode(d: &mut D) -> Rc<T> {
        Rc::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for [T] {
    default fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?
            }
            Ok(())
        })
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Vec<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        let slice: &[T] = self;
        slice.encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Vec<T> {
    default fn decode(d: &mut D) -> Vec<T> {
        d.read_seq(|d, len| {
            // SAFETY: we set the capacity in advance, only write elements, and
            // only set the length at the end once the writing has succeeded.
            let mut vec = Vec::with_capacity(len);
            unsafe {
                let ptr: *mut T = vec.as_mut_ptr();
                for i in 0..len {
                    std::ptr::write(
                        ptr.offset(i as isize),
                        d.read_seq_elt(|d| Decodable::decode(d)),
                    );
                }
                vec.set_len(len);
            }
            vec
        })
    }
}

impl<S: Encoder, T: Encodable<S>, const N: usize> Encodable<S> for [T; N] {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        let slice: &[T] = self;
        slice.encode(s)
    }
}

impl<D: Decoder, const N: usize> Decodable<D> for [u8; N] {
    fn decode(d: &mut D) -> [u8; N] {
        d.read_seq(|d, len| {
            assert!(len == N);
            let mut v = [0u8; N];
            for i in 0..len {
                v[i] = d.read_seq_elt(|d| Decodable::decode(d));
            }
            v
        })
    }
}

impl<'a, S: Encoder, T: Encodable<S>> Encodable<S> for Cow<'a, [T]>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        let slice: &[T] = self;
        slice.encode(s)
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

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Option<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_option(|s| match *self {
            None => s.emit_option_none(),
            Some(ref v) => s.emit_option_some(|s| v.encode(s)),
        })
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &mut D) -> Option<T> {
        d.read_option(|d, b| if b { Some(Decodable::decode(d)) } else { None })
    }
}

impl<S: Encoder, T1: Encodable<S>, T2: Encodable<S>> Encodable<S> for Result<T1, T2> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_enum(|s| match *self {
            Ok(ref v) => {
                s.emit_enum_variant("Ok", 0, 1, |s| s.emit_enum_variant_arg(true, |s| v.encode(s)))
            }
            Err(ref v) => {
                s.emit_enum_variant("Err", 1, 1, |s| s.emit_enum_variant_arg(true, |s| v.encode(s)))
            }
        })
    }
}

impl<D: Decoder, T1: Decodable<D>, T2: Decodable<D>> Decodable<D> for Result<T1, T2> {
    fn decode(d: &mut D) -> Result<T1, T2> {
        d.read_enum(|d| {
            d.read_enum_variant(&["Ok", "Err"], |d, disr| match disr {
                0 => Ok(d.read_enum_variant_arg(|d| T1::decode(d))),
                1 => Err(d.read_enum_variant_arg(|d| T2::decode(d))),
                _ => panic!("Encountered invalid discriminant while decoding `Result`."),
            })
        })
    }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

/// Evaluates to the number of tokens passed to it.
///
/// Logarithmic counting: every one or two recursive expansions, the number of
/// tokens to count is divided by two, instead of being reduced by one.
/// Therefore, the recursion depth is the binary logarithm of the number of
/// tokens to count, and the expanded tree is likewise very small.
macro_rules! count {
    ()                     => (0usize);
    ($one:tt)              => (1usize);
    ($($pairs:tt $_p:tt)*) => (count!($($pairs)*) << 1usize);
    ($odd:tt $($rest:tt)*) => (count!($($rest)*) | 1usize);
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        impl<D: Decoder, $($name: Decodable<D>),+> Decodable<D> for ($($name,)+) {
            #[allow(non_snake_case)]
            fn decode(d: &mut D) -> ($($name,)+) {
                let len: usize = count!($($name)+);
                d.read_tuple(len, |d| {
                    let ret = ($(d.read_tuple_arg(|d| -> $name {
                        Decodable::decode(d)
                    }),)+);
                    ret
                })
            }
        }
        impl<S: Encoder, $($name: Encodable<S>),+> Encodable<S> for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode(&self, s: &mut S) -> Result<(), S::Error> {
                let ($(ref $name,)+) = *self;
                let mut n = 0;
                $(let $name = $name; n += 1;)+
                s.emit_tuple(n, |s| {
                    let mut i = 0;
                    $(s.emit_tuple_arg({ i+=1; i-1 }, |s| $name.encode(s))?;)+
                    Ok(())
                })
            }
        }
        peel! { $($name,)+ }
    )
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

impl<S: Encoder> Encodable<S> for path::Path {
    fn encode(&self, e: &mut S) -> Result<(), S::Error> {
        self.to_str().unwrap().encode(e)
    }
}

impl<S: Encoder> Encodable<S> for path::PathBuf {
    fn encode(&self, e: &mut S) -> Result<(), S::Error> {
        path::Path::encode(self, e)
    }
}

impl<D: Decoder> Decodable<D> for path::PathBuf {
    fn decode(d: &mut D) -> path::PathBuf {
        let bytes: String = Decodable::decode(d);
        path::PathBuf::from(bytes)
    }
}

impl<S: Encoder, T: Encodable<S> + Copy> Encodable<S> for Cell<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        self.get().encode(s)
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
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        self.borrow().encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for RefCell<T> {
    fn decode(d: &mut D) -> RefCell<T> {
        RefCell::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for Arc<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Arc<T> {
    fn decode(d: &mut D) -> Arc<T> {
        Arc::new(Decodable::decode(d))
    }
}

impl<S: Encoder, T: ?Sized + Encodable<S>> Encodable<S> for Box<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}
impl<D: Decoder, T: Decodable<D>> Decodable<D> for Box<T> {
    fn decode(d: &mut D) -> Box<T> {
        Box::new(Decodable::decode(d))
    }
}
