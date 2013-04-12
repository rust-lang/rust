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

#[forbid(non_camel_case_types)];

use core::prelude::*;
use core::hashmap::{HashMap, HashSet};
use core::trie::{TrieMap, TrieSet};
use deque::Deque;
use dlist::DList;
use treemap::{TreeMap, TreeSet};

pub trait Encoder {
    // Primitive types:
    fn emit_nil(&self);
    fn emit_uint(&self, v: uint);
    fn emit_u64(&self, v: u64);
    fn emit_u32(&self, v: u32);
    fn emit_u16(&self, v: u16);
    fn emit_u8(&self, v: u8);
    fn emit_int(&self, v: int);
    fn emit_i64(&self, v: i64);
    fn emit_i32(&self, v: i32);
    fn emit_i16(&self, v: i16);
    fn emit_i8(&self, v: i8);
    fn emit_bool(&self, v: bool);
    fn emit_float(&self, v: float);
    fn emit_f64(&self, v: f64);
    fn emit_f32(&self, v: f32);
    fn emit_char(&self, v: char);
    fn emit_str(&self, v: &str);

    // Compound types:
    fn emit_enum(&self, name: &str, f: &fn());

    fn emit_enum_variant(&self, v_name: &str, v_id: uint, len: uint, f: &fn());
    fn emit_enum_variant_arg(&self, a_idx: uint, f: &fn());

    fn emit_enum_struct_variant(&self, v_name: &str, v_id: uint, len: uint, f: &fn());
    fn emit_enum_struct_variant_field(&self, f_name: &str, f_idx: uint, f: &fn());

    fn emit_struct(&self, name: &str, len: uint, f: &fn());
    #[cfg(stage0)]
    fn emit_field(&self, f_name: &str, f_idx: uint, f: &fn());
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn emit_struct_field(&self, f_name: &str, f_idx: uint, f: &fn());

    fn emit_tuple(&self, len: uint, f: &fn());
    fn emit_tuple_arg(&self, idx: uint, f: &fn());

    fn emit_tuple_struct(&self, name: &str, len: uint, f: &fn());
    fn emit_tuple_struct_arg(&self, f_idx: uint, f: &fn());

    // Specialized types:
    fn emit_option(&self, f: &fn());
    fn emit_option_none(&self);
    fn emit_option_some(&self, f: &fn());

    fn emit_seq(&self, len: uint, f: &fn());
    fn emit_seq_elt(&self, idx: uint, f: &fn());

    fn emit_map(&self, len: uint, f: &fn());
    fn emit_map_elt_key(&self, idx: uint, f: &fn());
    fn emit_map_elt_val(&self, idx: uint, f: &fn());
}

pub trait Decoder {
    // Primitive types:
    fn read_nil(&self) -> ();
    fn read_uint(&self) -> uint;
    fn read_u64(&self) -> u64;
    fn read_u32(&self) -> u32;
    fn read_u16(&self) -> u16;
    fn read_u8(&self) -> u8;
    fn read_int(&self) -> int;
    fn read_i64(&self) -> i64;
    fn read_i32(&self) -> i32;
    fn read_i16(&self) -> i16;
    fn read_i8(&self) -> i8;
    fn read_bool(&self) -> bool;
    fn read_f64(&self) -> f64;
    fn read_f32(&self) -> f32;
    fn read_float(&self) -> float;
    fn read_char(&self) -> char;
    fn read_str(&self) -> ~str;

    // Compound types:
    fn read_enum<T>(&self, name: &str, f: &fn() -> T) -> T;

    fn read_enum_variant<T>(&self, names: &[&str], f: &fn(uint) -> T) -> T;
    fn read_enum_variant_arg<T>(&self, a_idx: uint, f: &fn() -> T) -> T;

    fn read_enum_struct_variant<T>(&self, names: &[&str], f: &fn(uint) -> T) -> T;
    fn read_enum_struct_variant_field<T>(&self, &f_name: &str, f_idx: uint, f: &fn() -> T) -> T;

    fn read_struct<T>(&self, s_name: &str, len: uint, f: &fn() -> T) -> T;
    #[cfg(stage0)]
    fn read_field<T>(&self, f_name: &str, f_idx: uint, f: &fn() -> T) -> T;
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn read_struct_field<T>(&self, f_name: &str, f_idx: uint, f: &fn() -> T) -> T;

    fn read_tuple<T>(&self, f: &fn(uint) -> T) -> T;
    fn read_tuple_arg<T>(&self, a_idx: uint, f: &fn() -> T) -> T;

    fn read_tuple_struct<T>(&self, s_name: &str, f: &fn(uint) -> T) -> T;
    fn read_tuple_struct_arg<T>(&self, a_idx: uint, f: &fn() -> T) -> T;

    // Specialized types:
    fn read_option<T>(&self, f: &fn(bool) -> T) -> T;

    fn read_seq<T>(&self, f: &fn(uint) -> T) -> T;
    fn read_seq_elt<T>(&self, idx: uint, f: &fn() -> T) -> T;

    fn read_map<T>(&self, f: &fn(uint) -> T) -> T;
    fn read_map_elt_key<T>(&self, idx: uint, f: &fn() -> T) -> T;
    fn read_map_elt_val<T>(&self, idx: uint, f: &fn() -> T) -> T;
}

pub trait Encodable<E: Encoder> {
    fn encode(&self, e: &E);
}

pub trait Decodable<D:Decoder> {
    fn decode(d: &D) -> Self;
}

impl<E: Encoder> Encodable<E> for uint {
    fn encode(&self, e: &E) { e.emit_uint(*self) }
}

impl<D:Decoder> Decodable<D> for uint {
    fn decode(d: &D) -> uint {
        d.read_uint()
    }
}

impl<E: Encoder> Encodable<E> for u8 {
    fn encode(&self, e: &E) { e.emit_u8(*self) }
}

impl<D:Decoder> Decodable<D> for u8 {
    fn decode(d: &D) -> u8 {
        d.read_u8()
    }
}

impl<E: Encoder> Encodable<E> for u16 {
    fn encode(&self, e: &E) { e.emit_u16(*self) }
}

impl<D:Decoder> Decodable<D> for u16 {
    fn decode(d: &D) -> u16 {
        d.read_u16()
    }
}

impl<E: Encoder> Encodable<E> for u32 {
    fn encode(&self, e: &E) { e.emit_u32(*self) }
}

impl<D:Decoder> Decodable<D> for u32 {
    fn decode(d: &D) -> u32 {
        d.read_u32()
    }
}

impl<E: Encoder> Encodable<E> for u64 {
    fn encode(&self, e: &E) { e.emit_u64(*self) }
}

impl<D:Decoder> Decodable<D> for u64 {
    fn decode(d: &D) -> u64 {
        d.read_u64()
    }
}

impl<E: Encoder> Encodable<E> for int {
    fn encode(&self, e: &E) { e.emit_int(*self) }
}

impl<D:Decoder> Decodable<D> for int {
    fn decode(d: &D) -> int {
        d.read_int()
    }
}

impl<E: Encoder> Encodable<E> for i8 {
    fn encode(&self, e: &E) { e.emit_i8(*self) }
}

impl<D:Decoder> Decodable<D> for i8 {
    fn decode(d: &D) -> i8 {
        d.read_i8()
    }
}

impl<E: Encoder> Encodable<E> for i16 {
    fn encode(&self, e: &E) { e.emit_i16(*self) }
}

impl<D:Decoder> Decodable<D> for i16 {
    fn decode(d: &D) -> i16 {
        d.read_i16()
    }
}

impl<E: Encoder> Encodable<E> for i32 {
    fn encode(&self, e: &E) { e.emit_i32(*self) }
}

impl<D:Decoder> Decodable<D> for i32 {
    fn decode(d: &D) -> i32 {
        d.read_i32()
    }
}

impl<E: Encoder> Encodable<E> for i64 {
    fn encode(&self, e: &E) { e.emit_i64(*self) }
}

impl<D:Decoder> Decodable<D> for i64 {
    fn decode(d: &D) -> i64 {
        d.read_i64()
    }
}

impl<'self, E: Encoder> Encodable<E> for &'self str {
    fn encode(&self, e: &E) { e.emit_str(*self) }
}

impl<E: Encoder> Encodable<E> for ~str {
    fn encode(&self, e: &E) { e.emit_str(*self) }
}

impl<D:Decoder> Decodable<D> for ~str {
    fn decode(d: &D) -> ~str {
        d.read_str()
    }
}

impl<E: Encoder> Encodable<E> for @str {
    fn encode(&self, e: &E) { e.emit_str(*self) }
}

impl<D:Decoder> Decodable<D> for @str {
    fn decode(d: &D) -> @str { d.read_str().to_managed() }
}

impl<E: Encoder> Encodable<E> for float {
    fn encode(&self, e: &E) { e.emit_float(*self) }
}

impl<D:Decoder> Decodable<D> for float {
    fn decode(d: &D) -> float {
        d.read_float()
    }
}

impl<E: Encoder> Encodable<E> for f32 {
    fn encode(&self, e: &E) { e.emit_f32(*self) }
}

impl<D:Decoder> Decodable<D> for f32 {
    fn decode(d: &D) -> f32 {
        d.read_f32() }
}

impl<E: Encoder> Encodable<E> for f64 {
    fn encode(&self, e: &E) { e.emit_f64(*self) }
}

impl<D:Decoder> Decodable<D> for f64 {
    fn decode(d: &D) -> f64 {
        d.read_f64()
    }
}

impl<E: Encoder> Encodable<E> for bool {
    fn encode(&self, e: &E) { e.emit_bool(*self) }
}

impl<D:Decoder> Decodable<D> for bool {
    fn decode(d: &D) -> bool {
        d.read_bool()
    }
}

impl<E: Encoder> Encodable<E> for () {
    fn encode(&self, e: &E) { e.emit_nil() }
}

impl<D:Decoder> Decodable<D> for () {
    fn decode(d: &D) -> () {
        d.read_nil()
    }
}

impl<'self, E: Encoder,T:Encodable<E>> Encodable<E> for &'self T {
    fn encode(&self, e: &E) {
        (**self).encode(e)
    }
}

impl<E: Encoder,T:Encodable<E>> Encodable<E> for ~T {
    fn encode(&self, e: &E) {
        (**self).encode(e)
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for ~T {
    fn decode(d: &D) -> ~T {
        ~Decodable::decode(d)
    }
}

impl<E: Encoder,T:Encodable<E>> Encodable<E> for @T {
    fn encode(&self, e: &E) {
        (**self).encode(e)
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @T {
    fn decode(d: &D) -> @T {
        @Decodable::decode(d)
    }
}

impl<'self, E: Encoder,T:Encodable<E>> Encodable<E> for &'self [T] {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            for self.eachi |i, elt| {
                e.emit_seq_elt(i, || elt.encode(e))
            }
        }
    }
}

impl<E: Encoder,T:Encodable<E>> Encodable<E> for ~[T] {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            for self.eachi |i, elt| {
                e.emit_seq_elt(i, || elt.encode(e))
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for ~[T] {
    fn decode(d: &D) -> ~[T] {
        do d.read_seq |len| {
            do vec::from_fn(len) |i| {
                d.read_seq_elt(i, || Decodable::decode(d))
            }
        }
    }
}

impl<E: Encoder,T:Encodable<E>> Encodable<E> for @[T] {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            for self.eachi |i, elt| {
                e.emit_seq_elt(i, || elt.encode(e))
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @[T] {
    fn decode(d: &D) -> @[T] {
        do d.read_seq |len| {
            do at_vec::from_fn(len) |i| {
                d.read_seq_elt(i, || Decodable::decode(d))
            }
        }
    }
}

impl<E: Encoder,T:Encodable<E>> Encodable<E> for Option<T> {
    fn encode(&self, e: &E) {
        do e.emit_option {
            match *self {
                None => e.emit_option_none(),
                Some(ref v) => e.emit_option_some(|| v.encode(e)),
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Option<T> {
    fn decode(d: &D) -> Option<T> {
        do d.read_option |b| {
            if b {
                Some(Decodable::decode(d))
            } else {
                None
            }
        }
    }
}

impl<E: Encoder,T0:Encodable<E>,T1:Encodable<E>> Encodable<E> for (T0, T1) {
    fn encode(&self, e: &E) {
        match *self {
            (ref t0, ref t1) => {
                do e.emit_seq(2) {
                    e.emit_seq_elt(0, || t0.encode(e));
                    e.emit_seq_elt(1, || t1.encode(e));
                }
            }
        }
    }
}

impl<D:Decoder,T0:Decodable<D>,T1:Decodable<D>> Decodable<D> for (T0, T1) {
    fn decode(d: &D) -> (T0, T1) {
        do d.read_seq |len| {
            assert!(len == 2);
            (
                d.read_seq_elt(0, || Decodable::decode(d)),
                d.read_seq_elt(1, || Decodable::decode(d))
            )
        }
    }
}

impl<
    E: Encoder,
    T0: Encodable<E>,
    T1: Encodable<E>,
    T2: Encodable<E>
> Encodable<E> for (T0, T1, T2) {
    fn encode(&self, e: &E) {
        match *self {
            (ref t0, ref t1, ref t2) => {
                do e.emit_seq(3) {
                    e.emit_seq_elt(0, || t0.encode(e));
                    e.emit_seq_elt(1, || t1.encode(e));
                    e.emit_seq_elt(2, || t2.encode(e));
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
    fn decode(d: &D) -> (T0, T1, T2) {
        do d.read_seq |len| {
            assert!(len == 3);
            (
                d.read_seq_elt(0, || Decodable::decode(d)),
                d.read_seq_elt(1, || Decodable::decode(d)),
                d.read_seq_elt(2, || Decodable::decode(d))
            )
        }
    }
}

impl<
    E: Encoder,
    T0: Encodable<E>,
    T1: Encodable<E>,
    T2: Encodable<E>,
    T3: Encodable<E>
> Encodable<E> for (T0, T1, T2, T3) {
    fn encode(&self, e: &E) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3) => {
                do e.emit_seq(4) {
                    e.emit_seq_elt(0, || t0.encode(e));
                    e.emit_seq_elt(1, || t1.encode(e));
                    e.emit_seq_elt(2, || t2.encode(e));
                    e.emit_seq_elt(3, || t3.encode(e));
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
    fn decode(d: &D) -> (T0, T1, T2, T3) {
        do d.read_seq |len| {
            assert!(len == 4);
            (
                d.read_seq_elt(0, || Decodable::decode(d)),
                d.read_seq_elt(1, || Decodable::decode(d)),
                d.read_seq_elt(2, || Decodable::decode(d)),
                d.read_seq_elt(3, || Decodable::decode(d))
            )
        }
    }
}

impl<
    E: Encoder,
    T0: Encodable<E>,
    T1: Encodable<E>,
    T2: Encodable<E>,
    T3: Encodable<E>,
    T4: Encodable<E>
> Encodable<E> for (T0, T1, T2, T3, T4) {
    fn encode(&self, e: &E) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3, ref t4) => {
                do e.emit_seq(5) {
                    e.emit_seq_elt(0, || t0.encode(e));
                    e.emit_seq_elt(1, || t1.encode(e));
                    e.emit_seq_elt(2, || t2.encode(e));
                    e.emit_seq_elt(3, || t3.encode(e));
                    e.emit_seq_elt(4, || t4.encode(e));
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
    fn decode(d: &D)
      -> (T0, T1, T2, T3, T4) {
        do d.read_seq |len| {
            assert!(len == 5);
            (
                d.read_seq_elt(0, || Decodable::decode(d)),
                d.read_seq_elt(1, || Decodable::decode(d)),
                d.read_seq_elt(2, || Decodable::decode(d)),
                d.read_seq_elt(3, || Decodable::decode(d)),
                d.read_seq_elt(4, || Decodable::decode(d))
            )
        }
    }
}

impl<
    E: Encoder,
    T: Encodable<E> + Copy
> Encodable<E> for @mut DList<T> {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.size) {
            let mut i = 0;
            for self.each |elt| {
                e.emit_seq_elt(i, || elt.encode(e));
                i += 1;
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for @mut DList<T> {
    fn decode(d: &D) -> @mut DList<T> {
        let list = DList();
        do d.read_seq |len| {
            for uint::range(0, len) |i| {
                list.push(d.read_seq_elt(i, || Decodable::decode(d)));
            }
        }
        list
    }
}

impl<
    E: Encoder,
    T: Encodable<E>
> Encodable<E> for Deque<T> {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            for self.eachi |i, elt| {
                e.emit_seq_elt(i, || elt.encode(e));
            }
        }
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for Deque<T> {
    fn decode(d: &D) -> Deque<T> {
        let mut deque = Deque::new();
        do d.read_seq |len| {
            for uint::range(0, len) |i| {
                deque.add_back(d.read_seq_elt(i, || Decodable::decode(d)));
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
    fn encode(&self, e: &E) {
        do e.emit_map(self.len()) {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, || key.encode(e));
                e.emit_map_elt_val(i, || val.encode(e));
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
    fn decode(d: &D) -> HashMap<K, V> {
        do d.read_map |len| {
            let mut map = HashMap::with_capacity(len);
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, || Decodable::decode(d));
                let val = d.read_map_elt_val(i, || Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<
    E: Encoder,
    T: Encodable<E> + Hash + IterBytes + Eq
> Encodable<E> for HashSet<T> {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            let mut i = 0;
            for self.each |elt| {
                e.emit_seq_elt(i, || elt.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Hash + IterBytes + Eq
> Decodable<D> for HashSet<T> {
    fn decode(d: &D) -> HashSet<T> {
        do d.read_seq |len| {
            let mut set = HashSet::with_capacity(len);
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, || Decodable::decode(d)));
            }
            set
        }
    }
}

impl<
    E: Encoder,
    V: Encodable<E>
> Encodable<E> for TrieMap<V> {
    fn encode(&self, e: &E) {
        do e.emit_map(self.len()) {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, || key.encode(e));
                e.emit_map_elt_val(i, || val.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    V: Decodable<D>
> Decodable<D> for TrieMap<V> {
    fn decode(d: &D) -> TrieMap<V> {
        do d.read_map |len| {
            let mut map = TrieMap::new();
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, || Decodable::decode(d));
                let val = d.read_map_elt_val(i, || Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<E: Encoder> Encodable<E> for TrieSet {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            let mut i = 0;
            for self.each |elt| {
                e.emit_seq_elt(i, || elt.encode(e));
                i += 1;
            }
        }
    }
}

impl<D: Decoder> Decodable<D> for TrieSet {
    fn decode(d: &D) -> TrieSet {
        do d.read_seq |len| {
            let mut set = TrieSet::new();
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, || Decodable::decode(d)));
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
    fn encode(&self, e: &E) {
        do e.emit_map(self.len()) {
            let mut i = 0;
            for self.each |key, val| {
                e.emit_map_elt_key(i, || key.encode(e));
                e.emit_map_elt_val(i, || val.encode(e));
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
    fn decode(d: &D) -> TreeMap<K, V> {
        do d.read_map |len| {
            let mut map = TreeMap::new();
            for uint::range(0, len) |i| {
                let key = d.read_map_elt_key(i, || Decodable::decode(d));
                let val = d.read_map_elt_val(i, || Decodable::decode(d));
                map.insert(key, val);
            }
            map
        }
    }
}

impl<
    E: Encoder,
    T: Encodable<E> + Eq + TotalOrd
> Encodable<E> for TreeSet<T> {
    fn encode(&self, e: &E) {
        do e.emit_seq(self.len()) {
            let mut i = 0;
            for self.each |elt| {
                e.emit_seq_elt(i, || elt.encode(e));
                i += 1;
            }
        }
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Eq + TotalOrd
> Decodable<D> for TreeSet<T> {
    fn decode(d: &D) -> TreeSet<T> {
        do d.read_seq |len| {
            let mut set = TreeSet::new();
            for uint::range(0, len) |i| {
                set.insert(d.read_seq_elt(i, || Decodable::decode(d)));
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
    fn emit_from_vec<T>(&self, v: &[T], f: &fn(v: &T));
}

impl<E: Encoder> EncoderHelpers for E {
    fn emit_from_vec<T>(&self, v: &[T], f: &fn(v: &T)) {
        do self.emit_seq(v.len()) {
            for v.eachi |i, e| {
                do self.emit_seq_elt(i) {
                    f(e)
                }
            }
        }
    }
}

pub trait DecoderHelpers {
    fn read_to_vec<T>(&self, f: &fn() -> T) -> ~[T];
}

impl<D:Decoder> DecoderHelpers for D {
    fn read_to_vec<T>(&self, f: &fn() -> T) -> ~[T] {
        do self.read_seq |len| {
            do vec::from_fn(len) |i| {
                self.read_seq_elt(i, || f())
            }
        }
    }
}
