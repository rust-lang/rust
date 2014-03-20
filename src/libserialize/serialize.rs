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
use std::slice;

pub trait Encoder {
    // Primitive types:
    fn emit_nil(&mut self);
    fn emit_uint(&mut self, v: uint);
    fn emit_u64(&mut self, v: u64);
    fn emit_u32(&mut self, v: u32);
    fn emit_u16(&mut self, v: u16);
    fn emit_u8(&mut self, v: u8);
    fn emit_int(&mut self, v: int);
    fn emit_i64(&mut self, v: i64);
    fn emit_i32(&mut self, v: i32);
    fn emit_i16(&mut self, v: i16);
    fn emit_i8(&mut self, v: i8);
    fn emit_bool(&mut self, v: bool);
    fn emit_f64(&mut self, v: f64);
    fn emit_f32(&mut self, v: f32);
    fn emit_char(&mut self, v: char);
    fn emit_str(&mut self, v: &str);

    // Compound types:
    fn emit_enum(&mut self, name: &str, f: |&mut Self|);

    fn emit_enum_variant(&mut self,
                         v_name: &str,
                         v_id: uint,
                         len: uint,
                         f: |&mut Self|);
    fn emit_enum_variant_arg(&mut self, a_idx: uint, f: |&mut Self|);

    fn emit_enum_struct_variant(&mut self,
                                v_name: &str,
                                v_id: uint,
                                len: uint,
                                f: |&mut Self|);
    fn emit_enum_struct_variant_field(&mut self,
                                      f_name: &str,
                                      f_idx: uint,
                                      f: |&mut Self|);

    fn emit_struct(&mut self, name: &str, len: uint, f: |&mut Self|);
    fn emit_struct_field(&mut self,
                         f_name: &str,
                         f_idx: uint,
                         f: |&mut Self|);

    fn emit_tuple(&mut self, len: uint, f: |&mut Self|);
    fn emit_tuple_arg(&mut self, idx: uint, f: |&mut Self|);

    fn emit_tuple_struct(&mut self, name: &str, len: uint, f: |&mut Self|);
    fn emit_tuple_struct_arg(&mut self, f_idx: uint, f: |&mut Self|);

    // Specialized types:
    fn emit_option(&mut self, f: |&mut Self|);
    fn emit_option_none(&mut self);
    fn emit_option_some(&mut self, f: |&mut Self|);

    fn emit_seq(&mut self, len: uint, f: |this: &mut Self|);
    fn emit_seq_elt(&mut self, idx: uint, f: |this: &mut Self|);

    fn emit_map(&mut self, len: uint, f: |&mut Self|);
    fn emit_map_elt_key(&mut self, idx: uint, f: |&mut Self|);
    fn emit_map_elt_val(&mut self, idx: uint, f: |&mut Self|);
}

pub trait Decoder {
    // Primitive types:
    fn read_nil(&mut self) -> ();
    fn read_uint(&mut self) -> uint;
    fn read_u64(&mut self) -> u64;
    fn read_u32(&mut self) -> u32;
    fn read_u16(&mut self) -> u16;
    fn read_u8(&mut self) -> u8;
    fn read_int(&mut self) -> int;
    fn read_i64(&mut self) -> i64;
    fn read_i32(&mut self) -> i32;
    fn read_i16(&mut self) -> i16;
    fn read_i8(&mut self) -> i8;
    fn read_bool(&mut self) -> bool;
    fn read_f64(&mut self) -> f64;
    fn read_f32(&mut self) -> f32;
    fn read_char(&mut self) -> char;
    fn read_str(&mut self) -> ~str;

    // Compound types:
    fn read_enum<T>(&mut self, name: &str, f: |&mut Self| -> T) -> T;

    fn read_enum_variant<T>(&mut self,
                            names: &[&str],
                            f: |&mut Self, uint| -> T)
                            -> T;
    fn read_enum_variant_arg<T>(&mut self,
                                a_idx: uint,
                                f: |&mut Self| -> T)
                                -> T;

    fn read_enum_struct_variant<T>(&mut self,
                                   names: &[&str],
                                   f: |&mut Self, uint| -> T)
                                   -> T;
    fn read_enum_struct_variant_field<T>(&mut self,
                                         &f_name: &str,
                                         f_idx: uint,
                                         f: |&mut Self| -> T)
                                         -> T;

    fn read_struct<T>(&mut self, s_name: &str, len: uint, f: |&mut Self| -> T)
                      -> T;
    fn read_struct_field<T>(&mut self,
                            f_name: &str,
                            f_idx: uint,
                            f: |&mut Self| -> T)
                            -> T;

    fn read_tuple<T>(&mut self, f: |&mut Self, uint| -> T) -> T;
    fn read_tuple_arg<T>(&mut self, a_idx: uint, f: |&mut Self| -> T) -> T;

    fn read_tuple_struct<T>(&mut self,
                            s_name: &str,
                            f: |&mut Self, uint| -> T)
                            -> T;
    fn read_tuple_struct_arg<T>(&mut self,
                                a_idx: uint,
                                f: |&mut Self| -> T)
                                -> T;

    // Specialized types:
    fn read_option<T>(&mut self, f: |&mut Self, bool| -> T) -> T;

    fn read_seq<T>(&mut self, f: |&mut Self, uint| -> T) -> T;
    fn read_seq_elt<T>(&mut self, idx: uint, f: |&mut Self| -> T) -> T;

    fn read_map<T>(&mut self, f: |&mut Self, uint| -> T) -> T;
    fn read_map_elt_key<T>(&mut self, idx: uint, f: |&mut Self| -> T) -> T;
    fn read_map_elt_val<T>(&mut self, idx: uint, f: |&mut Self| -> T) -> T;
}

pub trait Encodable<S:Encoder> {
    fn encode(&self, s: &mut S);
}

pub trait Decodable<D:Decoder> {
    fn decode(d: &mut D) -> Self;
}

impl<S:Encoder> Encodable<S> for uint {
    fn encode(&self, s: &mut S) {
        s.emit_uint(*self)
    }
}

impl<D:Decoder> Decodable<D> for uint {
    fn decode(d: &mut D) -> uint {
        d.read_uint()
    }
}

impl<S:Encoder> Encodable<S> for u8 {
    fn encode(&self, s: &mut S) {
        s.emit_u8(*self)
    }
}

impl<D:Decoder> Decodable<D> for u8 {
    fn decode(d: &mut D) -> u8 {
        d.read_u8()
    }
}

impl<S:Encoder> Encodable<S> for u16 {
    fn encode(&self, s: &mut S) {
        s.emit_u16(*self)
    }
}

impl<D:Decoder> Decodable<D> for u16 {
    fn decode(d: &mut D) -> u16 {
        d.read_u16()
    }
}

impl<S:Encoder> Encodable<S> for u32 {
    fn encode(&self, s: &mut S) {
        s.emit_u32(*self)
    }
}

impl<D:Decoder> Decodable<D> for u32 {
    fn decode(d: &mut D) -> u32 {
        d.read_u32()
    }
}

impl<S:Encoder> Encodable<S> for u64 {
    fn encode(&self, s: &mut S) {
        s.emit_u64(*self)
    }
}

impl<D:Decoder> Decodable<D> for u64 {
    fn decode(d: &mut D) -> u64 {
        d.read_u64()
    }
}

impl<S:Encoder> Encodable<S> for int {
    fn encode(&self, s: &mut S) {
        s.emit_int(*self)
    }
}

impl<D:Decoder> Decodable<D> for int {
    fn decode(d: &mut D) -> int {
        d.read_int()
    }
}

impl<S:Encoder> Encodable<S> for i8 {
    fn encode(&self, s: &mut S) {
        s.emit_i8(*self)
    }
}

impl<D:Decoder> Decodable<D> for i8 {
    fn decode(d: &mut D) -> i8 {
        d.read_i8()
    }
}

impl<S:Encoder> Encodable<S> for i16 {
    fn encode(&self, s: &mut S) {
        s.emit_i16(*self)
    }
}

impl<D:Decoder> Decodable<D> for i16 {
    fn decode(d: &mut D) -> i16 {
        d.read_i16()
    }
}

impl<S:Encoder> Encodable<S> for i32 {
    fn encode(&self, s: &mut S) {
        s.emit_i32(*self)
    }
}

impl<D:Decoder> Decodable<D> for i32 {
    fn decode(d: &mut D) -> i32 {
        d.read_i32()
    }
}

impl<S:Encoder> Encodable<S> for i64 {
    fn encode(&self, s: &mut S) {
        s.emit_i64(*self)
    }
}

impl<D:Decoder> Decodable<D> for i64 {
    fn decode(d: &mut D) -> i64 {
        d.read_i64()
    }
}

impl<'a, S:Encoder> Encodable<S> for &'a str {
    fn encode(&self, s: &mut S) {
        s.emit_str(*self)
    }
}

impl<S:Encoder> Encodable<S> for ~str {
    fn encode(&self, s: &mut S) {
        s.emit_str(*self)
    }
}

impl<D:Decoder> Decodable<D> for ~str {
    fn decode(d: &mut D) -> ~str {
        d.read_str()
    }
}

impl<S:Encoder> Encodable<S> for f32 {
    fn encode(&self, s: &mut S) {
        s.emit_f32(*self)
    }
}

impl<D:Decoder> Decodable<D> for f32 {
    fn decode(d: &mut D) -> f32 {
        d.read_f32()
    }
}

impl<S:Encoder> Encodable<S> for f64 {
    fn encode(&self, s: &mut S) {
        s.emit_f64(*self)
    }
}

impl<D:Decoder> Decodable<D> for f64 {
    fn decode(d: &mut D) -> f64 {
        d.read_f64()
    }
}

impl<S:Encoder> Encodable<S> for bool {
    fn encode(&self, s: &mut S) {
        s.emit_bool(*self)
    }
}

impl<D:Decoder> Decodable<D> for bool {
    fn decode(d: &mut D) -> bool {
        d.read_bool()
    }
}

impl<S:Encoder> Encodable<S> for char {
    fn encode(&self, s: &mut S) {
        s.emit_char(*self)
    }
}

impl<D:Decoder> Decodable<D> for char {
    fn decode(d: &mut D) -> char {
        d.read_char()
    }
}

impl<S:Encoder> Encodable<S> for () {
    fn encode(&self, s: &mut S) {
        s.emit_nil()
    }
}

impl<D:Decoder> Decodable<D> for () {
    fn decode(d: &mut D) -> () {
        d.read_nil()
    }
}

impl<'a, S:Encoder,T:Encodable<S>> Encodable<S> for &'a T {
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for ~T {
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for ~T {
    fn decode(d: &mut D) -> ~T {
        ~Decodable::decode(d)
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for @T {
    fn encode(&self, s: &mut S) {
        (**self).encode(s)
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for Rc<T> {
    #[inline]
    fn encode(&self, s: &mut S) {
        self.deref().encode(s)
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Rc<T> {
    #[inline]
    fn decode(d: &mut D) -> Rc<T> {
        Rc::new(Decodable::decode(d))
    }
}

impl<D:Decoder,T:Decodable<D> + 'static> Decodable<D> for @T {
    fn decode(d: &mut D) -> @T {
        @Decodable::decode(d)
    }
}

impl<'a, S:Encoder,T:Encodable<S>> Encodable<S> for &'a [T] {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        })
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for ~[T] {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        })
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for ~[T] {
    fn decode(d: &mut D) -> ~[T] {
        d.read_seq(|d, len| {
            slice::from_fn(len, |i| {
                d.read_seq_elt(i, |d| Decodable::decode(d))
            })
        })
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for Vec<T> {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        })
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Vec<T> {
    fn decode(d: &mut D) -> Vec<T> {
        d.read_seq(|d, len| {
            Vec::from_fn(len, |i| {
                d.read_seq_elt(i, |d| Decodable::decode(d))
            })
        })
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for Option<T> {
    fn encode(&self, s: &mut S) {
        s.emit_option(|s| {
            match *self {
                None => s.emit_option_none(),
                Some(ref v) => s.emit_option_some(|s| v.encode(s)),
            }
        })
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &mut D) -> Option<T> {
        d.read_option(|d, b| {
            if b {
                Some(Decodable::decode(d))
            } else {
                None
            }
        })
    }
}

impl<S:Encoder,T0:Encodable<S>,T1:Encodable<S>> Encodable<S> for (T0, T1) {
    fn encode(&self, s: &mut S) {
        match *self {
            (ref t0, ref t1) => {
                s.emit_seq(2, |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                })
            }
        }
    }
}

impl<D:Decoder,T0:Decodable<D>,T1:Decodable<D>> Decodable<D> for (T0, T1) {
    fn decode(d: &mut D) -> (T0, T1) {
        d.read_seq(|d, len| {
            assert_eq!(len, 2);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d))
            )
        })
    }
}

impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>
> Encodable<S> for (T0, T1, T2) {
    fn encode(&self, s: &mut S) {
        match *self {
            (ref t0, ref t1, ref t2) => {
                s.emit_seq(3, |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                })
            }
        }
    }
}

impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>
> Decodable<D> for (T0, T1, T2) {
    fn decode(d: &mut D) -> (T0, T1, T2) {
        d.read_seq(|d, len| {
            assert_eq!(len, 3);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d))
            )
        })
    }
}

impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>,
    T3: Encodable<S>
> Encodable<S> for (T0, T1, T2, T3) {
    fn encode(&self, s: &mut S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3) => {
                s.emit_seq(4, |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                    s.emit_seq_elt(3, |s| t3.encode(s));
                })
            }
        }
    }
}

impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>,
    T3: Decodable<D>
> Decodable<D> for (T0, T1, T2, T3) {
    fn decode(d: &mut D) -> (T0, T1, T2, T3) {
        d.read_seq(|d, len| {
            assert_eq!(len, 4);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d)),
                d.read_seq_elt(3, |d| Decodable::decode(d))
            )
        })
    }
}

impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>,
    T3: Encodable<S>,
    T4: Encodable<S>
> Encodable<S> for (T0, T1, T2, T3, T4) {
    fn encode(&self, s: &mut S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3, ref t4) => {
                s.emit_seq(5, |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                    s.emit_seq_elt(3, |s| t3.encode(s));
                    s.emit_seq_elt(4, |s| t4.encode(s));
                })
            }
        }
    }
}

impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>,
    T3: Decodable<D>,
    T4: Decodable<D>
> Decodable<D> for (T0, T1, T2, T3, T4) {
    fn decode(d: &mut D) -> (T0, T1, T2, T3, T4) {
        d.read_seq(|d, len| {
            assert_eq!(len, 5);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d)),
                d.read_seq_elt(3, |d| Decodable::decode(d)),
                d.read_seq_elt(4, |d| Decodable::decode(d))
            )
        })
    }
}

impl<E: Encoder> Encodable<E> for path::posix::Path {
    fn encode(&self, e: &mut E) {
        self.as_vec().encode(e)
    }
}

impl<D: Decoder> Decodable<D> for path::posix::Path {
    fn decode(d: &mut D) -> path::posix::Path {
        let bytes: ~[u8] = Decodable::decode(d);
        path::posix::Path::new(bytes)
    }
}

impl<E: Encoder> Encodable<E> for path::windows::Path {
    fn encode(&self, e: &mut E) {
        self.as_vec().encode(e)
    }
}

impl<D: Decoder> Decodable<D> for path::windows::Path {
    fn decode(d: &mut D) -> path::windows::Path {
        let bytes: ~[u8] = Decodable::decode(d);
        path::windows::Path::new(bytes)
    }
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

pub trait EncoderHelpers {
    fn emit_from_vec<T>(&mut self, v: &[T], f: |&mut Self, v: &T|);
}

impl<S:Encoder> EncoderHelpers for S {
    fn emit_from_vec<T>(&mut self, v: &[T], f: |&mut S, &T|) {
        self.emit_seq(v.len(), |this| {
            for (i, e) in v.iter().enumerate() {
                this.emit_seq_elt(i, |this| {
                    f(this, e)
                })
            }
        })
    }
}

pub trait DecoderHelpers {
    fn read_to_vec<T>(&mut self, f: |&mut Self| -> T) -> ~[T];
}

impl<D:Decoder> DecoderHelpers for D {
    fn read_to_vec<T>(&mut self, f: |&mut D| -> T) -> ~[T] {
        self.read_seq(|this, len| {
            slice::from_fn(len, |i| {
                this.read_seq_elt(i, |this| f(this))
            })
        })
    }
}
