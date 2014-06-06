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

use std::path;
use std::rc::Rc;
use std::gc::Gc;

pub trait Encoder<E> {
    // Primitive types:
    fn emit_nil(&mut self) -> Result<(), E>;
    fn emit_uint(&mut self, v: uint) -> Result<(), E>;
    fn emit_u64(&mut self, v: u64) -> Result<(), E>;
    fn emit_u32(&mut self, v: u32) -> Result<(), E>;
    fn emit_u16(&mut self, v: u16) -> Result<(), E>;
    fn emit_u8(&mut self, v: u8) -> Result<(), E>;
    fn emit_int(&mut self, v: int) -> Result<(), E>;
    fn emit_i64(&mut self, v: i64) -> Result<(), E>;
    fn emit_i32(&mut self, v: i32) -> Result<(), E>;
    fn emit_i16(&mut self, v: i16) -> Result<(), E>;
    fn emit_i8(&mut self, v: i8) -> Result<(), E>;
    fn emit_bool(&mut self, v: bool) -> Result<(), E>;
    fn emit_f64(&mut self, v: f64) -> Result<(), E>;
    fn emit_f32(&mut self, v: f32) -> Result<(), E>;
    fn emit_char(&mut self, v: char) -> Result<(), E>;
    fn emit_str(&mut self, v: &str) -> Result<(), E>;

    // Compound types:
    fn emit_enum(&mut self, name: &str, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_enum_variant(&mut self,
                         v_name: &str,
                         v_id: uint,
                         len: uint,
                         f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_enum_variant_arg(&mut self,
                             a_idx: uint,
                             f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_enum_struct_variant(&mut self,
                                v_name: &str,
                                v_id: uint,
                                len: uint,
                                f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_enum_struct_variant_field(&mut self,
                                      f_name: &str,
                                      f_idx: uint,
                                      f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_struct(&mut self,
                   name: &str,
                   len: uint,
                   f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_struct_field(&mut self,
                         f_name: &str,
                         f_idx: uint,
                         f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_tuple(&mut self, len: uint, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_tuple_arg(&mut self, idx: uint, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_tuple_struct(&mut self,
                         name: &str,
                         len: uint,
                         f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_tuple_struct_arg(&mut self,
                             f_idx: uint,
                             f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    // Specialized types:
    fn emit_option(&mut self, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_option_none(&mut self) -> Result<(), E>;
    fn emit_option_some(&mut self, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_seq(&mut self, len: uint, f: |this: &mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_seq_elt(&mut self, idx: uint, f: |this: &mut Self| -> Result<(), E>) -> Result<(), E>;

    fn emit_map(&mut self, len: uint, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_map_elt_key(&mut self, idx: uint, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
    fn emit_map_elt_val(&mut self, idx: uint, f: |&mut Self| -> Result<(), E>) -> Result<(), E>;
}

pub trait Decoder<E> {
    // Primitive types:
    fn read_nil(&mut self) -> Result<(), E>;
    fn read_uint(&mut self) -> Result<uint, E>;
    fn read_u64(&mut self) -> Result<u64, E>;
    fn read_u32(&mut self) -> Result<u32, E>;
    fn read_u16(&mut self) -> Result<u16, E>;
    fn read_u8(&mut self) -> Result<u8, E>;
    fn read_int(&mut self) -> Result<int, E>;
    fn read_i64(&mut self) -> Result<i64, E>;
    fn read_i32(&mut self) -> Result<i32, E>;
    fn read_i16(&mut self) -> Result<i16, E>;
    fn read_i8(&mut self) -> Result<i8, E>;
    fn read_bool(&mut self) -> Result<bool, E>;
    fn read_f64(&mut self) -> Result<f64, E>;
    fn read_f32(&mut self) -> Result<f32, E>;
    fn read_char(&mut self) -> Result<char, E>;
    fn read_str(&mut self) -> Result<String, E>;

    // Compound types:
    fn read_enum<T>(&mut self, name: &str, f: |&mut Self| -> Result<T, E>) -> Result<T, E>;

    fn read_enum_variant<T>(&mut self,
                            names: &[&str],
                            f: |&mut Self, uint| -> Result<T, E>)
                            -> Result<T, E>;
    fn read_enum_variant_arg<T>(&mut self,
                                a_idx: uint,
                                f: |&mut Self| -> Result<T, E>)
                                -> Result<T, E>;

    fn read_enum_struct_variant<T>(&mut self,
                                   names: &[&str],
                                   f: |&mut Self, uint| -> Result<T, E>)
                                   -> Result<T, E>;
    fn read_enum_struct_variant_field<T>(&mut self,
                                         &f_name: &str,
                                         f_idx: uint,
                                         f: |&mut Self| -> Result<T, E>)
                                         -> Result<T, E>;

    fn read_struct<T>(&mut self, s_name: &str, len: uint, f: |&mut Self| -> Result<T, E>)
                      -> Result<T, E>;
    fn read_struct_field<T>(&mut self,
                            f_name: &str,
                            f_idx: uint,
                            f: |&mut Self| -> Result<T, E>)
                            -> Result<T, E>;

    fn read_tuple<T>(&mut self, f: |&mut Self, uint| -> Result<T, E>) -> Result<T, E>;
    fn read_tuple_arg<T>(&mut self, a_idx: uint, f: |&mut Self| -> Result<T, E>) -> Result<T, E>;

    fn read_tuple_struct<T>(&mut self,
                            s_name: &str,
                            f: |&mut Self, uint| -> Result<T, E>)
                            -> Result<T, E>;
    fn read_tuple_struct_arg<T>(&mut self,
                                a_idx: uint,
                                f: |&mut Self| -> Result<T, E>)
                                -> Result<T, E>;

    // Specialized types:
    fn read_option<T>(&mut self, f: |&mut Self, bool| -> Result<T, E>) -> Result<T, E>;

    fn read_seq<T>(&mut self, f: |&mut Self, uint| -> Result<T, E>) -> Result<T, E>;
    fn read_seq_elt<T>(&mut self, idx: uint, f: |&mut Self| -> Result<T, E>) -> Result<T, E>;

    fn read_map<T>(&mut self, f: |&mut Self, uint| -> Result<T, E>) -> Result<T, E>;
    fn read_map_elt_key<T>(&mut self, idx: uint, f: |&mut Self| -> Result<T, E>) -> Result<T, E>;
    fn read_map_elt_val<T>(&mut self, idx: uint, f: |&mut Self| -> Result<T, E>) -> Result<T, E>;
}

pub trait Encodable<S:Encoder<E>, E> {
    fn encode(&self, s: &mut S) -> Result<(), E>;
}

pub trait Decodable<D:Decoder<E>, E> {
    fn decode(d: &mut D) -> Result<Self, E>;
}

impl<E, S:Encoder<E>> Encodable<S, E> for uint {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_uint(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for uint {
    fn decode(d: &mut D) -> Result<uint, E> {
        d.read_uint()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for u8 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_u8(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for u8 {
    fn decode(d: &mut D) -> Result<u8, E> {
        d.read_u8()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for u16 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_u16(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for u16 {
    fn decode(d: &mut D) -> Result<u16, E> {
        d.read_u16()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for u32 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_u32(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for u32 {
    fn decode(d: &mut D) -> Result<u32, E> {
        d.read_u32()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for u64 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_u64(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for u64 {
    fn decode(d: &mut D) -> Result<u64, E> {
        d.read_u64()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for int {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_int(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for int {
    fn decode(d: &mut D) -> Result<int, E> {
        d.read_int()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for i8 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_i8(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for i8 {
    fn decode(d: &mut D) -> Result<i8, E> {
        d.read_i8()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for i16 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_i16(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for i16 {
    fn decode(d: &mut D) -> Result<i16, E> {
        d.read_i16()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for i32 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_i32(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for i32 {
    fn decode(d: &mut D) -> Result<i32, E> {
        d.read_i32()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for i64 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_i64(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for i64 {
    fn decode(d: &mut D) -> Result<i64, E> {
        d.read_i64()
    }
}

impl<'a, E, S:Encoder<E>> Encodable<S, E> for &'a str {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_str(*self)
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for String {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_str(self.as_slice())
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for String {
    fn decode(d: &mut D) -> Result<String, E> {
        Ok(String::from_str(try!(d.read_str()).as_slice()))
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for f32 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_f32(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for f32 {
    fn decode(d: &mut D) -> Result<f32, E> {
        d.read_f32()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for f64 {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_f64(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for f64 {
    fn decode(d: &mut D) -> Result<f64, E> {
        d.read_f64()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for bool {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_bool(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for bool {
    fn decode(d: &mut D) -> Result<bool, E> {
        d.read_bool()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for char {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_char(*self)
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for char {
    fn decode(d: &mut D) -> Result<char, E> {
        d.read_char()
    }
}

impl<E, S:Encoder<E>> Encodable<S, E> for () {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_nil()
    }
}

impl<E, D:Decoder<E>> Decodable<D, E> for () {
    fn decode(d: &mut D) -> Result<(), E> {
        d.read_nil()
    }
}

impl<'a, E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for &'a T {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        (**self).encode(s)
    }
}

impl<E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for Box<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        (**self).encode(s)
    }
}

impl<E, D:Decoder<E>,T:Decodable<D, E>> Decodable<D, E> for Box<T> {
    fn decode(d: &mut D) -> Result<Box<T>, E> {
        Ok(box try!(Decodable::decode(d)))
    }
}

impl<E, S:Encoder<E>,T:'static + Encodable<S, E>> Encodable<S, E> for Gc<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        (**self).encode(s)
    }
}

impl<E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for Rc<T> {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), E> {
        (**self).encode(s)
    }
}

impl<E, D:Decoder<E>,T:Decodable<D, E>> Decodable<D, E> for Rc<T> {
    #[inline]
    fn decode(d: &mut D) -> Result<Rc<T>, E> {
        Ok(Rc::new(try!(Decodable::decode(d))))
    }
}

impl<E, D:Decoder<E>,T:Decodable<D, E> + 'static> Decodable<D, E> for Gc<T> {
    fn decode(d: &mut D) -> Result<Gc<T>, E> {
        Ok(box(GC) try!(Decodable::decode(d)))
    }
}

impl<'a, E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for &'a [T] {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                try!(s.emit_seq_elt(i, |s| e.encode(s)))
            }
            Ok(())
        })
    }
}

impl<E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for Vec<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                try!(s.emit_seq_elt(i, |s| e.encode(s)))
            }
            Ok(())
        })
    }
}

impl<E, D:Decoder<E>,T:Decodable<D, E>> Decodable<D, E> for Vec<T> {
    fn decode(d: &mut D) -> Result<Vec<T>, E> {
        d.read_seq(|d, len| {
            let mut v = Vec::with_capacity(len);
            for i in range(0, len) {
                v.push(try!(d.read_seq_elt(i, |d| Decodable::decode(d))));
            }
            Ok(v)
        })
    }
}

impl<E, S:Encoder<E>,T:Encodable<S, E>> Encodable<S, E> for Option<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_option(|s| {
            match *self {
                None => s.emit_option_none(),
                Some(ref v) => s.emit_option_some(|s| v.encode(s)),
            }
        })
    }
}

impl<E, D:Decoder<E>,T:Decodable<D, E>> Decodable<D, E> for Option<T> {
    fn decode(d: &mut D) -> Result<Option<T>, E> {
        d.read_option(|d, b| {
            if b {
                Ok(Some(try!(Decodable::decode(d))))
            } else {
                Ok(None)
            }
        })
    }
}

macro_rules! peel(($name:ident, $($other:ident,)*) => (tuple!($($other,)*)))

macro_rules! tuple (
    () => ();
    ( $($name:ident,)+ ) => (
        impl<E, D:Decoder<E>,$($name:Decodable<D, E>),*> Decodable<D,E> for ($($name,)*) {
            #[allow(uppercase_variables)]
            fn decode(d: &mut D) -> Result<($($name,)*), E> {
                d.read_tuple(|d, amt| {
                    let mut i = 0;
                    let ret = ($(try!(d.read_tuple_arg({ i+=1; i-1 }, |d| -> Result<$name,E> {
                        Decodable::decode(d)
                    })),)*);
                    assert!(amt == i,
                            "expected tuple of length `{}`, found tuple \
                             of length `{}`", i, amt);
                    return Ok(ret);
                })
            }
        }
        impl<E, S:Encoder<E>,$($name:Encodable<S, E>),*> Encodable<S, E> for ($($name,)*) {
            #[allow(uppercase_variables)]
            fn encode(&self, s: &mut S) -> Result<(), E> {
                let ($(ref $name,)*) = *self;
                let mut n = 0;
                $(let $name = $name; n += 1;)*
                s.emit_tuple(n, |s| {
                    let mut i = 0;
                    $(try!(s.emit_seq_elt({ i+=1; i-1 }, |s| $name.encode(s)));)*
                    Ok(())
                })
            }
        }
        peel!($($name,)*)
    )
)

tuple! { T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, }

impl<E, S: Encoder<E>> Encodable<S, E> for path::posix::Path {
    fn encode(&self, e: &mut S) -> Result<(), E> {
        self.as_vec().encode(e)
    }
}

impl<E, D: Decoder<E>> Decodable<D, E> for path::posix::Path {
    fn decode(d: &mut D) -> Result<path::posix::Path, E> {
        let bytes: Vec<u8> = try!(Decodable::decode(d));
        Ok(path::posix::Path::new(bytes))
    }
}

impl<E, S: Encoder<E>> Encodable<S, E> for path::windows::Path {
    fn encode(&self, e: &mut S) -> Result<(), E> {
        self.as_vec().encode(e)
    }
}

impl<E, D: Decoder<E>> Decodable<D, E> for path::windows::Path {
    fn decode(d: &mut D) -> Result<path::windows::Path, E> {
        let bytes: Vec<u8> = try!(Decodable::decode(d));
        Ok(path::windows::Path::new(bytes))
    }
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

pub trait EncoderHelpers<E> {
    fn emit_from_vec<T>(&mut self,
                        v: &[T],
                        f: |&mut Self, v: &T| -> Result<(), E>) -> Result<(), E>;
}

impl<E, S:Encoder<E>> EncoderHelpers<E> for S {
    fn emit_from_vec<T>(&mut self, v: &[T], f: |&mut S, &T| -> Result<(), E>) -> Result<(), E> {
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

pub trait DecoderHelpers<E> {
    fn read_to_vec<T>(&mut self, f: |&mut Self| -> Result<T, E>) -> Result<Vec<T>, E>;
}

impl<E, D:Decoder<E>> DecoderHelpers<E> for D {
    fn read_to_vec<T>(&mut self, f: |&mut D| -> Result<T, E>) -> Result<Vec<T>, E> {
        self.read_seq(|this, len| {
            let mut v = Vec::with_capacity(len);
            for i in range(0, len) {
                v.push(try!(this.read_seq_elt(i, |this| f(this))));
            }
            Ok(v)
        })
    }
}
