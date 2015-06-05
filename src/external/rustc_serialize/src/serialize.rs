// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

use std::cell::{Cell, RefCell};
use std::ffi::OsString;
use std::path;
use std::rc::Rc;
use std::sync::Arc;
use std::marker::PhantomData;
use std::borrow::Cow;

pub trait Encoder {
    type Error;

    // Primitive types:
    fn emit_nil(&mut self) -> Result<(), Self::Error>;
    fn emit_usize(&mut self, v: usize) -> Result<(), Self::Error>;
    fn emit_u64(&mut self, v: u64) -> Result<(), Self::Error>;
    fn emit_u32(&mut self, v: u32) -> Result<(), Self::Error>;
    fn emit_u16(&mut self, v: u16) -> Result<(), Self::Error>;
    fn emit_u8(&mut self, v: u8) -> Result<(), Self::Error>;
    fn emit_isize(&mut self, v: isize) -> Result<(), Self::Error>;
    fn emit_i64(&mut self, v: i64) -> Result<(), Self::Error>;
    fn emit_i32(&mut self, v: i32) -> Result<(), Self::Error>;
    fn emit_i16(&mut self, v: i16) -> Result<(), Self::Error>;
    fn emit_i8(&mut self, v: i8) -> Result<(), Self::Error>;
    fn emit_bool(&mut self, v: bool) -> Result<(), Self::Error>;
    fn emit_f64(&mut self, v: f64) -> Result<(), Self::Error>;
    fn emit_f32(&mut self, v: f32) -> Result<(), Self::Error>;
    fn emit_char(&mut self, v: char) -> Result<(), Self::Error>;
    fn emit_str(&mut self, v: &str) -> Result<(), Self::Error>;

    // Compound types:
    fn emit_enum<F>(&mut self, name: &str, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_enum_variant<F>(&mut self, v_name: &str,
                            v_id: usize,
                            len: usize,
                            f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_enum_variant_arg<F>(&mut self, a_idx: usize, f: F)
                                -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_enum_struct_variant<F>(&mut self, v_name: &str,
                                   v_id: usize,
                                   len: usize,
                                   f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_enum_struct_variant_field<F>(&mut self,
                                         f_name: &str,
                                         f_idx: usize,
                                         f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_struct<F>(&mut self, name: &str, len: usize, f: F)
                      -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_struct_field<F>(&mut self, f_name: &str, f_idx: usize, f: F)
                            -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_tuple<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_tuple_struct<F>(&mut self, name: &str, len: usize, f: F)
                            -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_tuple_struct_arg<F>(&mut self, f_idx: usize, f: F)
                                -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    // Specialized types:
    fn emit_option<F>(&mut self, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_option_none(&mut self) -> Result<(), Self::Error>;
    fn emit_option_some<F>(&mut self, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_seq<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_seq_elt<F>(&mut self, idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;

    fn emit_map<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_map_elt_key<F>(&mut self, idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
    fn emit_map_elt_val<F>(&mut self, idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>;
}

pub trait Decoder {
    type Error;

    // Primitive types:
    fn read_nil(&mut self) -> Result<(), Self::Error>;
    fn read_usize(&mut self) -> Result<usize, Self::Error>;
    fn read_u64(&mut self) -> Result<u64, Self::Error>;
    fn read_u32(&mut self) -> Result<u32, Self::Error>;
    fn read_u16(&mut self) -> Result<u16, Self::Error>;
    fn read_u8(&mut self) -> Result<u8, Self::Error>;
    fn read_isize(&mut self) -> Result<isize, Self::Error>;
    fn read_i64(&mut self) -> Result<i64, Self::Error>;
    fn read_i32(&mut self) -> Result<i32, Self::Error>;
    fn read_i16(&mut self) -> Result<i16, Self::Error>;
    fn read_i8(&mut self) -> Result<i8, Self::Error>;
    fn read_bool(&mut self) -> Result<bool, Self::Error>;
    fn read_f64(&mut self) -> Result<f64, Self::Error>;
    fn read_f32(&mut self) -> Result<f32, Self::Error>;
    fn read_char(&mut self) -> Result<char, Self::Error>;
    fn read_str(&mut self) -> Result<String, Self::Error>;

    // Compound types:
    fn read_enum<T, F>(&mut self, name: &str, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_enum_variant<T, F>(&mut self, names: &[&str], f: F)
                               -> Result<T, Self::Error>
        where F: FnMut(&mut Self, usize) -> Result<T, Self::Error>;
    fn read_enum_variant_arg<T, F>(&mut self, a_idx: usize, f: F)
                                   -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_enum_struct_variant<T, F>(&mut self, names: &[&str], f: F)
                                      -> Result<T, Self::Error>
        where F: FnMut(&mut Self, usize) -> Result<T, Self::Error>;
    fn read_enum_struct_variant_field<T, F>(&mut self,
                                            &f_name: &str,
                                            f_idx: usize,
                                            f: F)
                                            -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_struct<T, F>(&mut self, s_name: &str, len: usize, f: F)
                         -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;
    fn read_struct_field<T, F>(&mut self,
                               f_name: &str,
                               f_idx: usize,
                               f: F)
                               -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_tuple<T, F>(&mut self, len: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;
    fn read_tuple_arg<T, F>(&mut self, a_idx: usize, f: F)
                            -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_tuple_struct<T, F>(&mut self, s_name: &str, len: usize, f: F)
                               -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;
    fn read_tuple_struct_arg<T, F>(&mut self, a_idx: usize, f: F)
                                   -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    // Specialized types:
    fn read_option<T, F>(&mut self, f: F) -> Result<T, Self::Error>
        where F: FnMut(&mut Self, bool) -> Result<T, Self::Error>;

    fn read_seq<T, F>(&mut self, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self, usize) -> Result<T, Self::Error>;
    fn read_seq_elt<T, F>(&mut self, idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    fn read_map<T, F>(&mut self, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self, usize) -> Result<T, Self::Error>;
    fn read_map_elt_key<T, F>(&mut self, idx: usize, f: F)
                              -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;
    fn read_map_elt_val<T, F>(&mut self, idx: usize, f: F)
                              -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;

    // Failure
    fn error(&mut self, err: &str) -> Self::Error;
}

pub trait Encodable {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error>;
}

pub trait Decodable {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error>;
}

impl Encodable for usize {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_usize(*self)
    }
}

impl Decodable for usize {
    fn decode<D: Decoder>(d: &mut D) -> Result<usize, D::Error> {
        d.read_usize()
    }
}

impl Encodable for u8 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u8(*self)
    }
}

impl Decodable for u8 {
    fn decode<D: Decoder>(d: &mut D) -> Result<u8, D::Error> {
        d.read_u8()
    }
}

impl Encodable for u16 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u16(*self)
    }
}

impl Decodable for u16 {
    fn decode<D: Decoder>(d: &mut D) -> Result<u16, D::Error> {
        d.read_u16()
    }
}

impl Encodable for u32 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(*self)
    }
}

impl Decodable for u32 {
    fn decode<D: Decoder>(d: &mut D) -> Result<u32, D::Error> {
        d.read_u32()
    }
}

impl Encodable for u64 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u64(*self)
    }
}

impl Decodable for u64 {
    fn decode<D: Decoder>(d: &mut D) -> Result<u64, D::Error> {
        d.read_u64()
    }
}

impl Encodable for isize {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_isize(*self)
    }
}

impl Decodable for isize {
    fn decode<D: Decoder>(d: &mut D) -> Result<isize, D::Error> {
        d.read_isize()
    }
}

impl Encodable for i8 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i8(*self)
    }
}

impl Decodable for i8 {
    fn decode<D: Decoder>(d: &mut D) -> Result<i8, D::Error> {
        d.read_i8()
    }
}

impl Encodable for i16 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i16(*self)
    }
}

impl Decodable for i16 {
    fn decode<D: Decoder>(d: &mut D) -> Result<i16, D::Error> {
        d.read_i16()
    }
}

impl Encodable for i32 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i32(*self)
    }
}

impl Decodable for i32 {
    fn decode<D: Decoder>(d: &mut D) -> Result<i32, D::Error> {
        d.read_i32()
    }
}

impl Encodable for i64 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i64(*self)
    }
}

impl Decodable for i64 {
    fn decode<D: Decoder>(d: &mut D) -> Result<i64, D::Error> {
        d.read_i64()
    }
}

impl Encodable for str {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self)
    }
}

impl Encodable for String {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self)
    }
}

impl Decodable for String {
    fn decode<D: Decoder>(d: &mut D) -> Result<String, D::Error> {
        d.read_str()
    }
}

impl Encodable for f32 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_f32(*self)
    }
}

impl Decodable for f32 {
    fn decode<D: Decoder>(d: &mut D) -> Result<f32, D::Error> {
        d.read_f32()
    }
}

impl Encodable for f64 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_f64(*self)
    }
}

impl Decodable for f64 {
    fn decode<D: Decoder>(d: &mut D) -> Result<f64, D::Error> {
        d.read_f64()
    }
}

impl Encodable for bool {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_bool(*self)
    }
}

impl Decodable for bool {
    fn decode<D: Decoder>(d: &mut D) -> Result<bool, D::Error> {
        d.read_bool()
    }
}

impl Encodable for char {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_char(*self)
    }
}

impl Decodable for char {
    fn decode<D: Decoder>(d: &mut D) -> Result<char, D::Error> {
        d.read_char()
    }
}

impl Encodable for () {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_nil()
    }
}

impl Decodable for () {
    fn decode<D: Decoder>(d: &mut D) -> Result<(), D::Error> {
        d.read_nil()
    }
}

impl<'a, T: ?Sized + Encodable> Encodable for &'a T {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T: ?Sized + Encodable> Encodable for Box<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl< T: Decodable> Decodable for Box<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Box<T>, D::Error> {
        Ok(Box::new(try!(Decodable::decode(d))))
    }
}

impl< T: Decodable> Decodable for Box<[T]> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Box<[T]>, D::Error> {
        let v: Vec<T> = try!(Decodable::decode(d));
        Ok(v.into_boxed_slice())
    }
}

impl<T:Encodable> Encodable for Rc<T> {
    #[inline]
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T:Decodable> Decodable for Rc<T> {
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<Rc<T>, D::Error> {
        Ok(Rc::new(try!(Decodable::decode(d))))
    }
}

impl<'a, T:Encodable + ToOwned + ?Sized> Encodable for Cow<'a, T> {
    #[inline]
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<'a, T: ?Sized> Decodable for Cow<'a, T>
    where T: ToOwned, T::Owned: Decodable
{
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<Cow<'static, T>, D::Error> {
        Ok(Cow::Owned(try!(Decodable::decode(d))))
    }
}

impl<T:Encodable> Encodable for [T] {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                try!(s.emit_seq_elt(i, |s| e.encode(s)))
            }
            Ok(())
        })
    }
}

impl<T:Encodable> Encodable for Vec<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                try!(s.emit_seq_elt(i, |s| e.encode(s)))
            }
            Ok(())
        })
    }
}

impl<T:Decodable> Decodable for Vec<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Vec<T>, D::Error> {
        d.read_seq(|d, len| {
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(try!(d.read_seq_elt(i, |d| Decodable::decode(d))));
            }
            Ok(v)
        })
    }
}

impl<T:Encodable> Encodable for Option<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_option(|s| {
            match *self {
                None => s.emit_option_none(),
                Some(ref v) => s.emit_option_some(|s| v.encode(s)),
            }
        })
    }
}

impl<T:Decodable> Decodable for Option<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Option<T>, D::Error> {
        d.read_option(|d, b| {
            if b {
                Ok(Some(try!(Decodable::decode(d))))
            } else {
                Ok(None)
            }
        })
    }
}

impl<T> Encodable for PhantomData<T> {
    fn encode<S: Encoder>(&self, _s: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}

impl<T> Decodable for PhantomData<T> {
    fn decode<D: Decoder>(_d: &mut D) -> Result<PhantomData<T>, D::Error> {
        Ok(PhantomData)
    }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

/// Evaluates to the number of identifiers passed to it, for example:
/// `count_idents!(a, b, c) == 3
macro_rules! count_idents {
    () => { 0 };
    ($_i:ident, $($rest:ident,)*) => { 1 + count_idents!($($rest,)*) }
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        impl<$($name:Decodable),*> Decodable for ($($name,)*) {
            fn decode<D: Decoder>(d: &mut D) -> Result<($($name,)*), D::Error> {
                let len: usize = count_idents!($($name,)*);
                d.read_tuple(len, |d| {
                    let mut i = 0;
                    let ret = ($(try!(d.read_tuple_arg({ i+=1; i-1 },
                                                       |d| -> Result<$name,D::Error> {
                        Decodable::decode(d)
                    })),)*);
                    return Ok(ret);
                })
            }
        }
        impl<$($name:Encodable),*> Encodable for ($($name,)*) {
            #[allow(non_snake_case)]
            fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
                let ($(ref $name,)*) = *self;
                let mut n = 0;
                $(let $name = $name; n += 1;)*
                s.emit_tuple(n, |s| {
                    let mut i = 0;
                    $(try!(s.emit_tuple_arg({ i+=1; i-1 }, |s| $name.encode(s)));)*
                    Ok(())
                })
            }
        }
        peel! { $($name,)* }
    )
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

macro_rules! array {
    ($zero:expr) => ();
    ($len:expr, $($idx:expr),*) => {
        impl<T:Decodable> Decodable for [T; $len] {
            fn decode<D: Decoder>(d: &mut D) -> Result<[T; $len], D::Error> {
                d.read_seq(|d, len| {
                    if len != $len {
                        return Err(d.error("wrong array length"));
                    }
                    Ok([$(
                        try!(d.read_seq_elt($len - $idx - 1,
                                            |d| Decodable::decode(d)))
                    ),+])
                })
            }
        }

        impl<T:Encodable> Encodable for [T; $len] {
            fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
                s.emit_seq($len, |s| {
                    for i in 0..$len {
                        try!(s.emit_seq_elt(i, |s| self[i].encode(s)));
                    }
                    Ok(())
                })
            }
        }
        array! { $($idx),* }
    }
}

array! {
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
}

impl Encodable for path::Path {
    #[cfg(unix)]
    fn encode<S: Encoder>(&self, e: &mut S) -> Result<(), S::Error> {
        use std::os::unix::prelude::*;
        self.as_os_str().as_bytes().encode(e)
    }
    #[cfg(windows)]
    fn encode<S: Encoder>(&self, e: &mut S) -> Result<(), S::Error> {
        use std::os::windows::prelude::*;
        let v = self.as_os_str().encode_wide().collect::<Vec<_>>();
        v.encode(e)
    }
}

impl Encodable for path::PathBuf {
    fn encode<S: Encoder>(&self, e: &mut S) -> Result<(), S::Error> {
        (**self).encode(e)
    }
}

impl Decodable for path::PathBuf {
    #[cfg(unix)]
    fn decode<D: Decoder>(d: &mut D) -> Result<path::PathBuf, D::Error> {
        use std::os::unix::prelude::*;
        let bytes: Vec<u8> = try!(Decodable::decode(d));
        let s: OsString = OsStringExt::from_vec(bytes);
        let mut p = path::PathBuf::new();
        p.push(s);
        Ok(p)
    }
    #[cfg(windows)]
    fn decode<D: Decoder>(d: &mut D) -> Result<path::PathBuf, D::Error> {
        use std::os::windows::prelude::*;
        let bytes: Vec<u16> = try!(Decodable::decode(d));
        let s: OsString = OsStringExt::from_wide(&bytes);
        let mut p = path::PathBuf::new();
        p.push(s);
        Ok(p)
    }
}

impl<T: Encodable + Copy> Encodable for Cell<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.get().encode(s)
    }
}

impl<T: Decodable + Copy> Decodable for Cell<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Cell<T>, D::Error> {
        Ok(Cell::new(try!(Decodable::decode(d))))
    }
}

// FIXME: #15036
// Should use `try_borrow`, returning a
// `encoder.error("attempting to Encode borrowed RefCell")`
// from `encode` when `try_borrow` returns `None`.

impl<T: Encodable> Encodable for RefCell<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.borrow().encode(s)
    }
}

impl<T: Decodable> Decodable for RefCell<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<RefCell<T>, D::Error> {
        Ok(RefCell::new(try!(Decodable::decode(d))))
    }
}

impl<T:Encodable> Encodable for Arc<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T:Decodable+Send+Sync> Decodable for Arc<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Arc<T>, D::Error> {
        Ok(Arc::new(try!(Decodable::decode(d))))
    }
}

// ___________________________________________________________________________
// Helper routines

pub trait EncoderHelpers: Encoder {
    fn emit_from_vec<T, F>(&mut self, v: &[T], f: F)
                           -> Result<(), <Self as Encoder>::Error>
        where F: FnMut(&mut Self, &T) -> Result<(), <Self as Encoder>::Error>;
}

impl<S:Encoder> EncoderHelpers for S {
    fn emit_from_vec<T, F>(&mut self, v: &[T], mut f: F) -> Result<(), S::Error> where
        F: FnMut(&mut S, &T) -> Result<(), S::Error>,
    {
        self.emit_seq(v.len(), |this| {
            for (i, e) in v.iter().enumerate() {
                try!(this.emit_seq_elt(i, |this| {
                    f(this, e)
                }));
            }
            Ok(())
        })
    }
}

pub trait DecoderHelpers: Decoder {
    fn read_to_vec<T, F>(&mut self, f: F)
                         -> Result<Vec<T>, <Self as Decoder>::Error> where
        F: FnMut(&mut Self) -> Result<T, <Self as Decoder>::Error>;
}

impl<D: Decoder> DecoderHelpers for D {
    fn read_to_vec<T, F>(&mut self, mut f: F) -> Result<Vec<T>, D::Error> where F:
        FnMut(&mut D) -> Result<T, D::Error>,
    {
        self.read_seq(|this, len| {
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(try!(this.read_seq_elt(i, |this| f(this))));
            }
            Ok(v)
        })
    }
}
