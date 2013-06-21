// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

#[allow(missing_doc)];
#[forbid(non_camel_case_types)];

use core::prelude::*;

use core::at_vec;
use core::hashmap::{HashMap, HashSet};
use core::trie::{TrieMap, TrieSet};
use core::uint;
use core::vec;
use deque::Deque;
use dlist::DList;
use treemap::{TreeMap, TreeSet};

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
    fn emit_float(&mut self, v: float);
    fn emit_f64(&mut self, v: f64);
    fn emit_f32(&mut self, v: f32);
    fn emit_char(&mut self, v: char);
    fn emit_str(&mut self, v: &str);

    // Compound types:
    fn emit_enum(&mut self, name: &str, f: &fn(&mut Self));

    fn emit_enum_variant(&mut self,
                         v_name: &str,
                         v_id: uint,
                         len: uint,
                         f: &fn(&mut Self));
    fn emit_enum_variant_arg(&mut self, a_idx: uint, f: &fn(&mut Self));

    fn emit_enum_struct_variant(&mut self,
                                v_name: &str,
                                v_id: uint,
                                len: uint,
                                f: &fn(&mut Self));
    fn emit_enum_struct_variant_field(&mut self,
                                      f_name: &str,
                                      f_idx: uint,
                                      f: &fn(&mut Self));

    fn emit_struct(&mut self, name: &str, len: uint, f: &fn(&mut Self));
    fn emit_struct_field(&mut self,
                         f_name: &str,
                         f_idx: uint,
                         f: &fn(&mut Self));

    fn emit_tuple(&mut self, len: uint, f: &fn(&mut Self));
    fn emit_tuple_arg(&mut self, idx: uint, f: &fn(&mut Self));

    fn emit_tuple_struct(&mut self, name: &str, len: uint, f: &fn(&mut Self));
    fn emit_tuple_struct_arg(&mut self, f_idx: uint, f: &fn(&mut Self));

    // Specialized types:
    fn emit_option(&mut self, f: &fn(&mut Self));
    fn emit_option_none(&mut self);
    fn emit_option_some(&mut self, f: &fn(&mut Self));

    fn emit_seq(&mut self, len: uint, f: &fn(this: &mut Self));
    fn emit_seq_elt(&mut self, idx: uint, f: &fn(this: &mut Self));

    fn emit_map(&mut self, len: uint, f: &fn(&mut Self));
    fn emit_map_elt_key(&mut self, idx: uint, f: &fn(&mut Self));
    fn emit_map_elt_val(&mut self, idx: uint, f: &fn(&mut Self));
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
    fn read_float(&mut self) -> float;
    fn read_char(&mut self) -> char;
    fn read_str(&mut self) -> ~str;

    // Compound types:
    fn read_enum<T>(&mut self, name: &str, f: &fn(&mut Self) -> T) -> T;

    fn read_enum_variant<T>(&mut self,
                            names: &[&str],
                            f: &fn(&mut Self, uint) -> T)
                            -> T;
    fn read_enum_variant_arg<T>(&mut self,
                                a_idx: uint,
                                f: &fn(&mut Self) -> T)
                                -> T;

    fn read_enum_struct_variant<T>(&mut self,
                                   names: &[&str],
                                   f: &fn(&mut Self, uint) -> T)
                                   -> T;
    fn read_enum_struct_variant_field<T>(&mut self,
                                         &f_name: &str,
                                         f_idx: uint,
                                         f: &fn(&mut Self) -> T)
                                         -> T;

    fn read_struct<T>(&mut self,
                      s_name: &str,
                      len: uint,
                      f: &fn(&mut Self) -> T)
                      -> T;
    fn read_struct_field<T>(&mut self,
                            f_name: &str,
                            f_idx: uint,
                            f: &fn(&mut Self) -> T)
                            -> T;

    fn read_tuple<T>(&mut self, f: &fn(&mut Self, uint) -> T) -> T;
    fn read_tuple_arg<T>(&mut self, a_idx: uint, f: &fn(&mut Self) -> T) -> T;

    fn read_tuple_struct<T>(&mut self,
                            s_name: &str,
                            f: &fn(&mut Self, uint) -> T)
                            -> T;
    fn read_tuple_struct_arg<T>(&mut self,
                                a_idx: uint,
                                f: &fn(&mut Self) -> T)
                                -> T;

    // Specialized types:
    fn read_option<T>(&mut self, f: &fn(&mut Self, bool) -> T) -> T;

    fn read_seq<T>(&mut self, f: &fn(&mut Self, uint) -> T) -> T;
    fn read_seq_elt<T>(&mut self, idx: uint, f: &fn(&mut Self) -> T) -> T;

    fn read_map<T>(&mut self, f: &fn(&mut Self, uint) -> T) -> T;
    fn read_map_elt_key<T>(&mut self, idx: uint, f: &fn(&mut Self) -> T) -> T;
    fn read_map_elt_val<T>(&mut self, idx: uint, f: &fn(&mut Self) -> T) -> T;
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

impl<'self, S:Encoder> Encodable<S> for &'self str {
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

impl<S:Encoder> Encodable<S> for @str {
    fn encode(&self, s: &mut S) {
        s.emit_str(*self)
    }
}

impl<D:Decoder> Decodable<D> for @str {
    fn decode(d: &mut D) -> @str {
        d.read_str().to_managed()
    }
}

impl<S:Encoder> Encodable<S> for float {
    fn encode(&self, s: &mut S) {
        s.emit_float(*self)
    }
}

impl<D:Decoder> Decodable<D> for float {
    fn decode(d: &mut D) -> float {
        d.read_float()
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

impl<'self, S:Encoder,T:Encodable<S>> Encodable<S> for &'self T {
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

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @T {
    fn decode(d: &mut D) -> @T {
        @Decodable::decode(d)
    }
}

impl<'self, S:Encoder,T:Encodable<S>> Encodable<S> for &'self [T] {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            for self.iter().enumerate().advance |(i, e)| {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        }
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for ~[T] {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            for self.iter().enumerate().advance |(i, e)| {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for ~[T] {
    fn decode(d: &mut D) -> ~[T] {
        do d.read_seq |d, len| {
            do vec::from_fn(len) |i| {
                d.read_seq_elt(i, |d| Decodable::decode(d))
            }
        }
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for @[T] {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            for self.iter().enumerate().advance |(i, e)| {
                s.emit_seq_elt(i, |s| e.encode(s))
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @[T] {
    fn decode(d: &mut D) -> @[T] {
        do d.read_seq |d, len| {
            do at_vec::from_fn(len) |i| {
                d.read_seq_elt(i, |d| Decodable::decode(d))
            }
        }
    }
}

impl<S:Encoder,T:Encodable<S>> Encodable<S> for Option<T> {
    fn encode(&self, s: &mut S) {
        do s.emit_option |s| {
            match *self {
                None => s.emit_option_none(),
                Some(ref v) => s.emit_option_some(|s| v.encode(s)),
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &mut D) -> Option<T> {
        do d.read_option |d, b| {
            if b {
                Some(Decodable::decode(d))
            } else {
                None
            }
        }
    }
}

impl<S:Encoder,T0:Encodable<S>,T1:Encodable<S>> Encodable<S> for (T0, T1) {
    fn encode(&self, s: &mut S) {
        match *self {
            (ref t0, ref t1) => {
                do s.emit_seq(2) |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                }
            }
        }
    }
}

impl<D:Decoder,T0:Decodable<D>,T1:Decodable<D>> Decodable<D> for (T0, T1) {
    fn decode(d: &mut D) -> (T0, T1) {
        do d.read_seq |d, len| {
            assert_eq!(len, 2);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d))
            )
        }
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
                do s.emit_seq(3) |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                }
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
        do d.read_seq |d, len| {
            assert_eq!(len, 3);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d))
            )
        }
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
                do s.emit_seq(4) |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                    s.emit_seq_elt(3, |s| t3.encode(s));
                }
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
        do d.read_seq |d, len| {
            assert_eq!(len, 4);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d)),
                d.read_seq_elt(3, |d| Decodable::decode(d))
            )
        }
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
                do s.emit_seq(5) |s| {
                    s.emit_seq_elt(0, |s| t0.encode(s));
                    s.emit_seq_elt(1, |s| t1.encode(s));
                    s.emit_seq_elt(2, |s| t2.encode(s));
                    s.emit_seq_elt(3, |s| t3.encode(s));
                    s.emit_seq_elt(4, |s| t4.encode(s));
                }
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
        do d.read_seq |d, len| {
            assert_eq!(len, 5);
            (
                d.read_seq_elt(0, |d| Decodable::decode(d)),
                d.read_seq_elt(1, |d| Decodable::decode(d)),
                d.read_seq_elt(2, |d| Decodable::decode(d)),
                d.read_seq_elt(3, |d| Decodable::decode(d)),
                d.read_seq_elt(4, |d| Decodable::decode(d))
            )
        }
    }
}

impl<
    S: Encoder,
    T: Encodable<S> + Copy
> Encodable<S> for @mut DList<T> {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.size) |s| {
            let mut i = 0;
            for self.each |e| {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @mut DList<T> {
    fn decode(d: &mut D) -> @mut DList<T> {
        let list = DList();
        do d.read_seq |d, len| {
            for uint::range(0, len) |i| {
                list.push(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
        }
        list
    }
}

impl<
    S: Encoder,
    T: Encodable<S>
> Encodable<S> for Deque<T> {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            for self.eachi |i, e| {
                s.emit_seq_elt(i, |s| e.encode(s));
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Deque<T> {
    fn decode(d: &mut D) -> Deque<T> {
        let mut deque = Deque::new();
        do d.read_seq |d, len| {
            for uint::range(0, len) |i| {
                deque.add_back(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
        }
        deque
    }
}

impl<
    E: Encoder,
    K: Encodable<E> + Hash + IterBytes + Eq,
    V: Encodable<E>
> Encodable<E> for HashMap<K, V> {
    fn encode(&self, e: &mut E) {
        do e.emit_map(self.len()) |e| {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, |e| key.encode(e));
                e.emit_map_elt_val(i, |e| val.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    K: Decodable<D> + Hash + IterBytes + Eq,
    V: Decodable<D>
> Decodable<D> for HashMap<K, V> {
    fn decode(d: &mut D) -> HashMap<K, V> {
        do d.read_map |d, len| {
            let mut map = HashMap::with_capacity(len);
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<
    S: Encoder,
    T: Encodable<S> + Hash + IterBytes + Eq
> Encodable<S> for HashSet<T> {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            let mut i = 0;
            for self.each |e| {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Hash + IterBytes + Eq
> Decodable<D> for HashSet<T> {
    fn decode(d: &mut D) -> HashSet<T> {
        do d.read_seq |d, len| {
            let mut set = HashSet::with_capacity(len);
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        }
    }
}

impl<
    E: Encoder,
    V: Encodable<E>
> Encodable<E> for TrieMap<V> {
    fn encode(&self, e: &mut E) {
        do e.emit_map(self.len()) |e| {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, |e| key.encode(e));
                e.emit_map_elt_val(i, |e| val.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    V: Decodable<D>
> Decodable<D> for TrieMap<V> {
    fn decode(d: &mut D) -> TrieMap<V> {
        do d.read_map |d, len| {
            let mut map = TrieMap::new();
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<S: Encoder> Encodable<S> for TrieSet {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            let mut i = 0;
            for self.each |e| {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        }
    }
}

impl<D: Decoder> Decodable<D> for TrieSet {
    fn decode(d: &mut D) -> TrieSet {
        do d.read_seq |d, len| {
            let mut set = TrieSet::new();
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        }
    }
}

impl<
    E: Encoder,
    K: Encodable<E> + Eq + TotalOrd,
    V: Encodable<E> + Eq
> Encodable<E> for TreeMap<K, V> {
    fn encode(&self, e: &mut E) {
        do e.emit_map(self.len()) |e| {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, |e| key.encode(e));
                e.emit_map_elt_val(i, |e| val.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    K: Decodable<D> + Eq + TotalOrd,
    V: Decodable<D> + Eq
> Decodable<D> for TreeMap<K, V> {
    fn decode(d: &mut D) -> TreeMap<K, V> {
        do d.read_map |d, len| {
            let mut map = TreeMap::new();
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<
    S: Encoder,
    T: Encodable<S> + Eq + TotalOrd
> Encodable<S> for TreeSet<T> {
    fn encode(&self, s: &mut S) {
        do s.emit_seq(self.len()) |s| {
            let mut i = 0;
            for self.each |e| {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Eq + TotalOrd
> Decodable<D> for TreeSet<T> {
    fn decode(d: &mut D) -> TreeSet<T> {
        do d.read_seq |d, len| {
            let mut set = TreeSet::new();
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        }
    }
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

pub trait EncoderHelpers {
    fn emit_from_vec<T>(&mut self, v: &[T], f: &fn(&mut Self, v: &T));
}

impl<S:Encoder> EncoderHelpers for S {
    fn emit_from_vec<T>(&mut self, v: &[T], f: &fn(&mut S, &T)) {
        do self.emit_seq(v.len()) |this| {
            for v.iter().enumerate().advance |(i, e)| {
                do this.emit_seq_elt(i) |this| {
                    f(this, e)
                }
            }
        }
    }
}

pub trait DecoderHelpers {
    fn read_to_vec<T>(&mut self, f: &fn(&mut Self) -> T) -> ~[T];
}

impl<D:Decoder> DecoderHelpers for D {
    fn read_to_vec<T>(&mut self, f: &fn(&mut D) -> T) -> ~[T] {
        do self.read_seq |this, len| {
            do vec::from_fn(len) |i| {
                this.read_seq_elt(i, |this| f(this))
            }
        }
    }
}
