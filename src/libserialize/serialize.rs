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

use std::borrow::Cow;
use std::intrinsics;
use std::path;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

pub trait Encoder {
    type Error;

    // Primitive types:
    fn emit_nil(&mut self) -> Result<(), Self::Error>;
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

    // Compound types:
    fn emit_enum<F>(&mut self, _name: &str, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_enum_variant<F>(&mut self, _v_name: &str, v_id: usize, _len: usize, f: F)
        -> Result<(), Self::Error> where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_usize(v_id)?;
        f(self)
    }

    fn emit_enum_variant_arg<F>(&mut self, _a_idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_enum_struct_variant<F>(&mut self, v_name: &str, v_id: usize, len: usize, f: F)
        -> Result<(), Self::Error> where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_enum_variant(v_name, v_id, len, f)
    }

    fn emit_enum_struct_variant_field<F>(&mut self, _f_name: &str, f_idx: usize, f: F)
        -> Result<(), Self::Error> where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_enum_variant_arg(f_idx, f)
    }

    fn emit_struct<F>(&mut self, _name: &str, _len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_struct_field<F>(&mut self, _f_name: &str, _f_idx: usize, f: F)
        -> Result<(), Self::Error> where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_tuple<F>(&mut self, _len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_tuple_arg<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_tuple_struct<F>(&mut self, _name: &str, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_tuple(len, f)
    }

    fn emit_tuple_struct_arg<F>(&mut self, f_idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_tuple_arg(f_idx, f)
    }

    // Specialized types:
    fn emit_option<F>(&mut self, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_enum("Option", f)
    }

    #[inline]
    fn emit_option_none(&mut self) -> Result<(), Self::Error> {
        self.emit_enum_variant("None", 0, 0, |_| Ok(()))
    }

    fn emit_option_some<F>(&mut self, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_enum_variant("Some", 1, 1, f)
    }

    fn emit_seq<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_usize(len)?;
        f(self)
    }

    fn emit_seq_elt<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_map<F>(&mut self, len: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        self.emit_usize(len)?;
        f(self)
    }

    fn emit_map_elt_key<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }

    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> Result<(), Self::Error>
        where F: FnOnce(&mut Self) -> Result<(), Self::Error>
    {
        f(self)
    }
}

pub trait Decoder {
    type Error;

    // Primitive types:
    fn read_nil(&mut self) -> Result<(), Self::Error>;
    fn read_usize(&mut self) -> Result<usize, Self::Error>;
    fn read_u128(&mut self) -> Result<u128, Self::Error>;
    fn read_u64(&mut self) -> Result<u64, Self::Error>;
    fn read_u32(&mut self) -> Result<u32, Self::Error>;
    fn read_u16(&mut self) -> Result<u16, Self::Error>;
    fn read_u8(&mut self) -> Result<u8, Self::Error>;
    fn read_isize(&mut self) -> Result<isize, Self::Error>;
    fn read_i128(&mut self) -> Result<i128, Self::Error>;
    fn read_i64(&mut self) -> Result<i64, Self::Error>;
    fn read_i32(&mut self) -> Result<i32, Self::Error>;
    fn read_i16(&mut self) -> Result<i16, Self::Error>;
    fn read_i8(&mut self) -> Result<i8, Self::Error>;
    fn read_bool(&mut self) -> Result<bool, Self::Error>;
    fn read_f64(&mut self) -> Result<f64, Self::Error>;
    fn read_f32(&mut self) -> Result<f32, Self::Error>;
    fn read_char(&mut self) -> Result<char, Self::Error>;
    fn read_str(&mut self) -> Result<Cow<str>, Self::Error>;

    // Compound types:
    fn read_enum<T, F>(&mut self, _name: &str, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_enum_variant<T, F>(&mut self, _names: &[&str], mut f: F) -> Result<T, Self::Error>
        where F: FnMut(&mut Self, usize) -> Result<T, Self::Error>
    {
        let disr = self.read_usize()?;
        f(self, disr)
    }

    fn read_enum_variant_arg<T, F>(&mut self, _a_idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_enum_struct_variant<T, F>(&mut self, names: &[&str], f: F) -> Result<T, Self::Error>
        where F: FnMut(&mut Self, usize) -> Result<T, Self::Error>
    {
        self.read_enum_variant(names, f)
    }

    fn read_enum_struct_variant_field<T, F>(&mut self, _f_name: &str, f_idx: usize, f: F)
        -> Result<T, Self::Error> where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        self.read_enum_variant_arg(f_idx, f)
    }

    fn read_struct<T, F>(&mut self, _s_name: &str, _len: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_struct_field<T, F>(&mut self, _f_name: &str, _f_idx: usize, f: F)
        -> Result<T, Self::Error> where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_tuple<T, F>(&mut self, _len: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_tuple_arg<T, F>(&mut self, _a_idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_tuple_struct<T, F>(&mut self, _s_name: &str, len: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        self.read_tuple(len, f)
    }

    fn read_tuple_struct_arg<T, F>(&mut self, a_idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        self.read_tuple_arg(a_idx, f)
    }

    // Specialized types:
    fn read_option<T, F>(&mut self, mut f: F) -> Result<T, Self::Error>
        where F: FnMut(&mut Self, bool) -> Result<T, Self::Error>
    {
        self.read_enum("Option", move |this| {
            this.read_enum_variant(&["None", "Some"], move |this, idx| {
                match idx {
                    0 => f(this, false),
                    1 => f(this, true),
                    _ => Err(this.error("read_option: expected 0 for None or 1 for Some")),
                }
            })
        })
    }

    fn read_seq<T, F>(&mut self, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self, usize) -> Result<T, Self::Error>
    {
        let len = self.read_usize()?;
        f(self, len)
    }

    fn read_seq_elt<T, F>(&mut self, _idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_map<T, F>(&mut self, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self, usize) -> Result<T, Self::Error>
    {
        let len = self.read_usize()?;
        f(self, len)
    }

    fn read_map_elt_key<T, F>(&mut self, _idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    fn read_map_elt_val<T, F>(&mut self, _idx: usize, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>
    {
        f(self)
    }

    // Failure
    fn error(&mut self, err: &str) -> Self::Error;
}

pub trait Encodable {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error>;
}

pub trait Decodable: Sized {
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

impl Encodable for u128 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u128(*self)
    }
}

impl Decodable for u128 {
    fn decode<D: Decoder>(d: &mut D) -> Result<u128, D::Error> {
        d.read_u128()
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

impl Encodable for i128 {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_i128(*self)
    }
}

impl Decodable for i128 {
    fn decode<D: Decoder>(d: &mut D) -> Result<i128, D::Error> {
        d.read_i128()
    }
}

impl Encodable for str {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self)
    }
}

impl Encodable for String {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self[..])
    }
}

impl Decodable for String {
    fn decode<D: Decoder>(d: &mut D) -> Result<String, D::Error> {
        Ok(d.read_str()?.into_owned())
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
        Ok(box Decodable::decode(d)?)
    }
}

impl< T: Decodable> Decodable for Box<[T]> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Box<[T]>, D::Error> {
        let v: Vec<T> = Decodable::decode(d)?;
        Ok(v.into_boxed_slice())
    }
}

impl<T:Encodable> Encodable for Rc<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T:Decodable> Decodable for Rc<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Rc<T>, D::Error> {
        Ok(Rc::new(Decodable::decode(d)?))
    }
}

impl<T:Encodable> Encodable for [T] {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?
            }
            Ok(())
        })
    }
}

impl<T:Encodable> Encodable for Vec<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?
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
                v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
            }
            Ok(v)
        })
    }
}

impl<'a, T:Encodable> Encodable for Cow<'a, [T]> where [T]: ToOwned<Owned = Vec<T>> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?
            }
            Ok(())
        })
    }
}

impl<T:Decodable+ToOwned> Decodable for Cow<'static, [T]>
    where [T]: ToOwned<Owned = Vec<T>>
{
    fn decode<D: Decoder>(d: &mut D) -> Result<Cow<'static, [T]>, D::Error> {
        d.read_seq(|d, len| {
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(d.read_seq_elt(i, |d| Decodable::decode(d))?);
            }
            Ok(Cow::Owned(v))
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
                Ok(Some(Decodable::decode(d)?))
            } else {
                Ok(None)
            }
        })
    }
}

impl<T1: Encodable, T2: Encodable> Encodable for Result<T1, T2> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_enum("Result", |s| {
            match *self {
                Ok(ref v) => {
                    s.emit_enum_variant("Ok", 0, 1, |s| {
                        s.emit_enum_variant_arg(0, |s| {
                            v.encode(s)
                        })
                    })
                }
                Err(ref v) => {
                    s.emit_enum_variant("Err", 1, 1, |s| {
                        s.emit_enum_variant_arg(0, |s| {
                            v.encode(s)
                        })
                    })
                }
            }
        })
    }
}

impl<T1:Decodable, T2:Decodable> Decodable for Result<T1, T2> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Result<T1, T2>, D::Error> {
        d.read_enum("Result", |d| {
            d.read_enum_variant(&["Ok", "Err"], |d, disr| {
                match disr {
                    0 => {
                        Ok(Ok(d.read_enum_variant_arg(0, |d| {
                            T1::decode(d)
                        })?))
                    }
                    1 => {
                        Ok(Err(d.read_enum_variant_arg(0, |d| {
                            T2::decode(d)
                        })?))
                    }
                    _ => {
                        panic!("Encountered invalid discriminant while \
                                decoding `Result`.");
                    }
                }
            })
        })
    }
}

macro_rules! peel {
    ($name:ident, $($other:ident,)*) => (tuple! { $($other,)* })
}

/// Evaluates to the number of identifiers passed to it, for example: `count_idents!(a, b, c) == 3
macro_rules! count_idents {
    () => { 0 };
    ($_i:ident, $($rest:ident,)*) => { 1 + count_idents!($($rest,)*) }
}

macro_rules! tuple {
    () => ();
    ( $($name:ident,)+ ) => (
        impl<$($name:Decodable),*> Decodable for ($($name,)*) {
            #[allow(non_snake_case)]
            fn decode<D: Decoder>(d: &mut D) -> Result<($($name,)*), D::Error> {
                let len: usize = count_idents!($($name,)*);
                d.read_tuple(len, |d| {
                    let mut i = 0;
                    let ret = ($(d.read_tuple_arg({ i+=1; i-1 }, |d| -> Result<$name, D::Error> {
                        Decodable::decode(d)
                    })?,)*);
                    Ok(ret)
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
                    $(s.emit_tuple_arg({ i+=1; i-1 }, |s| $name.encode(s))?;)*
                    Ok(())
                })
            }
        }
        peel! { $($name,)* }
    )
}

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

impl Encodable for path::PathBuf {
    fn encode<S: Encoder>(&self, e: &mut S) -> Result<(), S::Error> {
        self.to_str().unwrap().encode(e)
    }
}

impl Decodable for path::PathBuf {
    fn decode<D: Decoder>(d: &mut D) -> Result<path::PathBuf, D::Error> {
        let bytes: String = Decodable::decode(d)?;
        Ok(path::PathBuf::from(bytes))
    }
}

impl<T: Encodable + Copy> Encodable for Cell<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.get().encode(s)
    }
}

impl<T: Decodable + Copy> Decodable for Cell<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Cell<T>, D::Error> {
        Ok(Cell::new(Decodable::decode(d)?))
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
        Ok(RefCell::new(Decodable::decode(d)?))
    }
}

impl<T:Encodable> Encodable for Arc<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T:Decodable> Decodable for Arc<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Arc<T>, D::Error> {
        Ok(Arc::new(Decodable::decode(d)?))
    }
}

// ___________________________________________________________________________
// Specialization-based interface for multi-dispatch Encodable/Decodable.

/// Implement this trait on your `{Encodable,Decodable}::Error` types
/// to override the default panic behavior for missing specializations.
pub trait SpecializationError {
    /// Create an error for a missing method specialization.
    /// Defaults to panicking with type, trait & method names.
    /// `S` is the encoder/decoder state type,
    /// `T` is the type being encoded/decoded, and
    /// the arguments are the names of the trait
    /// and method that should've been overridden.
    fn not_found<S, T: ?Sized>(trait_name: &'static str, method_name: &'static str) -> Self;
}

impl<E> SpecializationError for E {
    default fn not_found<S, T: ?Sized>(trait_name: &'static str, method_name: &'static str) -> E {
        panic!("missing specialization: `<{} as {}<{}>>::{}` not overridden",
               unsafe { intrinsics::type_name::<S>() },
               trait_name,
               unsafe { intrinsics::type_name::<T>() },
               method_name);
    }
}

/// Implement this trait on encoders, with `T` being the type
/// you want to encode (employing `UseSpecializedEncodable`),
/// using a strategy specific to the encoder.
pub trait SpecializedEncoder<T: ?Sized + UseSpecializedEncodable>: Encoder {
    /// Encode the value in a manner specific to this encoder state.
    fn specialized_encode(&mut self, value: &T) -> Result<(), Self::Error>;
}

impl<E: Encoder, T: ?Sized + UseSpecializedEncodable> SpecializedEncoder<T> for E {
    default fn specialized_encode(&mut self, value: &T) -> Result<(), E::Error> {
        value.default_encode(self)
    }
}

/// Implement this trait on decoders, with `T` being the type
/// you want to decode (employing `UseSpecializedDecodable`),
/// using a strategy specific to the decoder.
pub trait SpecializedDecoder<T: UseSpecializedDecodable>: Decoder {
    /// Decode a value in a manner specific to this decoder state.
    fn specialized_decode(&mut self) -> Result<T, Self::Error>;
}

impl<D: Decoder, T: UseSpecializedDecodable> SpecializedDecoder<T> for D {
    default fn specialized_decode(&mut self) -> Result<T, D::Error> {
        T::default_decode(self)
    }
}

/// Implement this trait on your type to get an `Encodable`
/// implementation which goes through `SpecializedEncoder`.
pub trait UseSpecializedEncodable {
    /// Defaults to returning an error (see `SpecializationError`).
    fn default_encode<E: Encoder>(&self, _: &mut E) -> Result<(), E::Error> {
        Err(E::Error::not_found::<E, Self>("SpecializedEncoder", "specialized_encode"))
    }
}

impl<T: ?Sized + UseSpecializedEncodable> Encodable for T {
    default fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        E::specialized_encode(e, self)
    }
}

/// Implement this trait on your type to get an `Decodable`
/// implementation which goes through `SpecializedDecoder`.
pub trait UseSpecializedDecodable: Sized {
    /// Defaults to returning an error (see `SpecializationError`).
    fn default_decode<D: Decoder>(_: &mut D) -> Result<Self, D::Error> {
        Err(D::Error::not_found::<D, Self>("SpecializedDecoder", "specialized_decode"))
    }
}

impl<T: UseSpecializedDecodable> Decodable for T {
    default fn decode<D: Decoder>(d: &mut D) -> Result<T, D::Error> {
        D::specialized_decode(d)
    }
}

// Can't avoid specialization for &T and Box<T> impls,
// as proxy impls on them are blankets that conflict
// with the Encodable and Decodable impls above,
// which only have `default` on their methods
// for this exact reason.
// May be fixable in a simpler fashion via the
// more complex lattice model for specialization.
impl<'a, T: ?Sized + Encodable> UseSpecializedEncodable for &'a T {}
impl<T: ?Sized + Encodable> UseSpecializedEncodable for Box<T> {}
impl<T: Decodable> UseSpecializedDecodable for Box<T> {}
