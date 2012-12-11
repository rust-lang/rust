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

#[forbid(deprecated_mode)];
#[forbid(non_camel_case_types)];

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
    fn emit_borrowed_str(&self, v: &str);
    fn emit_owned_str(&self, v: &str);
    fn emit_managed_str(&self, v: &str);

    // Compound types:
    fn emit_borrowed(&self, f: fn());
    fn emit_owned(&self, f: fn());
    fn emit_managed(&self, f: fn());

    fn emit_enum(&self, name: &str, f: fn());
    fn emit_enum_variant(&self, v_name: &str, v_id: uint, sz: uint, f: fn());
    fn emit_enum_variant_arg(&self, idx: uint, f: fn());

    fn emit_borrowed_vec(&self, len: uint, f: fn());
    fn emit_owned_vec(&self, len: uint, f: fn());
    fn emit_managed_vec(&self, len: uint, f: fn());
    fn emit_vec_elt(&self, idx: uint, f: fn());

    fn emit_rec(&self, f: fn());
    fn emit_struct(&self, name: &str, f: fn());
    fn emit_field(&self, f_name: &str, f_idx: uint, f: fn());

    fn emit_tup(&self, len: uint, f: fn());
    fn emit_tup_elt(&self, idx: uint, f: fn());
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
    fn read_owned_str(&self) -> ~str;
    fn read_managed_str(&self) -> @str;

    // Compound types:
    fn read_enum<T>(&self, name: &str, f: fn() -> T) -> T;
    fn read_enum_variant<T>(&self, f: fn(uint) -> T) -> T;
    fn read_enum_variant_arg<T>(&self, idx: uint, f: fn() -> T) -> T;

    fn read_owned<T>(&self, f: fn() -> T) -> T;
    fn read_managed<T>(&self, f: fn() -> T) -> T;

    fn read_owned_vec<T>(&self, f: fn(uint) -> T) -> T;
    fn read_managed_vec<T>(&self, f: fn(uint) -> T) -> T;
    fn read_vec_elt<T>(&self, idx: uint, f: fn() -> T) -> T;

    fn read_rec<T>(&self, f: fn() -> T) -> T;
    fn read_struct<T>(&self, name: &str, f: fn() -> T) -> T;
    fn read_field<T>(&self, name: &str, idx: uint, f: fn() -> T) -> T;

    fn read_tup<T>(&self, sz: uint, f: fn() -> T) -> T;
    fn read_tup_elt<T>(&self, idx: uint, f: fn() -> T) -> T;
}

pub mod traits {
pub trait Encodable<S: Encoder> {
    fn encode(&self, s: &S);
}

pub trait Decodable<D: Decoder> {
    static fn decode(&self, d: &D) -> self;
}

pub impl<S: Encoder> uint: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_uint(*self) }
}

pub impl<D: Decoder> uint: Decodable<D> {
    static fn decode(&self, d: &D) -> uint {
        d.read_uint()
    }
}

pub impl<S: Encoder> u8: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_u8(*self) }
}

pub impl<D: Decoder> u8: Decodable<D> {
    static fn decode(&self, d: &D) -> u8 {
        d.read_u8()
    }
}

pub impl<S: Encoder> u16: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_u16(*self) }
}

pub impl<D: Decoder> u16: Decodable<D> {
    static fn decode(&self, d: &D) -> u16 {
        d.read_u16()
    }
}

pub impl<S: Encoder> u32: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_u32(*self) }
}

pub impl<D: Decoder> u32: Decodable<D> {
    static fn decode(&self, d: &D) -> u32 {
        d.read_u32()
    }
}

pub impl<S: Encoder> u64: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_u64(*self) }
}

pub impl<D: Decoder> u64: Decodable<D> {
    static fn decode(&self, d: &D) -> u64 {
        d.read_u64()
    }
}

pub impl<S: Encoder> int: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_int(*self) }
}

pub impl<D: Decoder> int: Decodable<D> {
    static fn decode(&self, d: &D) -> int {
        d.read_int()
    }
}

pub impl<S: Encoder> i8: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_i8(*self) }
}

pub impl<D: Decoder> i8: Decodable<D> {
    static fn decode(&self, d: &D) -> i8 {
        d.read_i8()
    }
}

pub impl<S: Encoder> i16: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_i16(*self) }
}

pub impl<D: Decoder> i16: Decodable<D> {
    static fn decode(&self, d: &D) -> i16 {
        d.read_i16()
    }
}

pub impl<S: Encoder> i32: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_i32(*self) }
}

pub impl<D: Decoder> i32: Decodable<D> {
    static fn decode(&self, d: &D) -> i32 {
        d.read_i32()
    }
}

pub impl<S: Encoder> i64: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_i64(*self) }
}

pub impl<D: Decoder> i64: Decodable<D> {
    static fn decode(&self, d: &D) -> i64 {
        d.read_i64()
    }
}

pub impl<S: Encoder> &str: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_borrowed_str(*self) }
}

pub impl<S: Encoder> ~str: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_owned_str(*self) }
}

pub impl<D: Decoder> ~str: Decodable<D> {
    static fn decode(&self, d: &D) -> ~str {
        d.read_owned_str()
    }
}

pub impl<S: Encoder> @str: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_managed_str(*self) }
}

pub impl<D: Decoder> @str: Decodable<D> {
    static fn decode(&self, d: &D) -> @str {
        d.read_managed_str()
    }
}

pub impl<S: Encoder> float: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_float(*self) }
}

pub impl<D: Decoder> float: Decodable<D> {
    static fn decode(&self, d: &D) -> float {
        d.read_float()
    }
}

pub impl<S: Encoder> f32: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_f32(*self) }
}

pub impl<D: Decoder> f32: Decodable<D> {
    static fn decode(&self, d: &D) -> f32 {
        d.read_f32() }
}

pub impl<S: Encoder> f64: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_f64(*self) }
}

pub impl<D: Decoder> f64: Decodable<D> {
    static fn decode(&self, d: &D) -> f64 {
        d.read_f64()
    }
}

pub impl<S: Encoder> bool: Encodable<S> {
    fn encode(&self, s: &S) { s.emit_bool(*self) }
}

pub impl<D: Decoder> bool: Decodable<D> {
    static fn decode(&self, d: &D) -> bool {
        d.read_bool()
    }
}

pub impl<S: Encoder> (): Encodable<S> {
    fn encode(&self, s: &S) { s.emit_nil() }
}

pub impl<D: Decoder> (): Decodable<D> {
    static fn decode(&self, d: &D) -> () {
        d.read_nil()
    }
}

pub impl<S: Encoder, T: Encodable<S>> &T: Encodable<S> {
    fn encode(&self, s: &S) {
        s.emit_borrowed(|| (**self).encode(s))
    }
}

pub impl<S: Encoder, T: Encodable<S>> ~T: Encodable<S> {
    fn encode(&self, s: &S) {
        s.emit_owned(|| (**self).encode(s))
    }
}

pub impl<D: Decoder, T: Decodable<D>> ~T: Decodable<D> {
    static fn decode(&self, d: &D) -> ~T {
        d.read_owned(|| ~decode(d))
    }
}

pub impl<S: Encoder, T: Encodable<S>> @T: Encodable<S> {
    fn encode(&self, s: &S) {
        s.emit_managed(|| (**self).encode(s))
    }
}

pub impl<D: Decoder, T: Decodable<D>> @T: Decodable<D> {
    static fn decode(&self, d: &D) -> @T {
        d.read_managed(|| @decode(d))
    }
}

pub impl<S: Encoder, T: Encodable<S>> &[T]: Encodable<S> {
    fn encode(&self, s: &S) {
        do s.emit_borrowed_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.encode(s))
            }
        }
    }
}

pub impl<S: Encoder, T: Encodable<S>> ~[T]: Encodable<S> {
    fn encode(&self, s: &S) {
        do s.emit_owned_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.encode(s))
            }
        }
    }
}

pub impl<D: Decoder, T: Decodable<D>> ~[T]: Decodable<D> {
    static fn decode(&self, d: &D) -> ~[T] {
        do d.read_owned_vec |len| {
            do vec::from_fn(len) |i| {
                d.read_vec_elt(i, || decode(d))
            }
        }
    }
}

pub impl<S: Encoder, T: Encodable<S>> @[T]: Encodable<S> {
    fn encode(&self, s: &S) {
        do s.emit_managed_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.encode(s))
            }
        }
    }
}

pub impl<D: Decoder, T: Decodable<D>> @[T]: Decodable<D> {
    static fn decode(&self, d: &D) -> @[T] {
        do d.read_managed_vec |len| {
            do at_vec::from_fn(len) |i| {
                d.read_vec_elt(i, || decode(d))
            }
        }
    }
}

pub impl<S: Encoder, T: Encodable<S>> Option<T>: Encodable<S> {
    fn encode(&self, s: &S) {
        do s.emit_enum(~"option") {
            match *self {
              None => do s.emit_enum_variant(~"none", 0u, 0u) {
              },

              Some(ref v) => do s.emit_enum_variant(~"some", 1u, 1u) {
                s.emit_enum_variant_arg(0u, || v.encode(s))
              }
            }
        }
    }
}

pub impl<D: Decoder, T: Decodable<D>> Option<T>: Decodable<D> {
    static fn decode(&self, d: &D) -> Option<T> {
        do d.read_enum(~"option") {
            do d.read_enum_variant |i| {
                match i {
                  0 => None,
                  1 => Some(d.read_enum_variant_arg(0u, || decode(d))),
                  _ => fail(fmt!("Bad variant for option: %u", i))
                }
            }
        }
    }
}

pub impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>
> (T0, T1): Encodable<S> {
    fn encode(&self, s: &S) {
        match *self {
            (ref t0, ref t1) => {
                do s.emit_tup(2) {
                    s.emit_tup_elt(0, || t0.encode(s));
                    s.emit_tup_elt(1, || t1.encode(s));
                }
            }
        }
    }
}

pub impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>
> (T0, T1): Decodable<D> {
    static fn decode(&self, d: &D) -> (T0, T1) {
        do d.read_tup(2) {
            (
                d.read_tup_elt(0, || decode(d)),
                d.read_tup_elt(1, || decode(d))
            )
        }
    }
}

pub impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>
> (T0, T1, T2): Encodable<S> {
    fn encode(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2) => {
                do s.emit_tup(3) {
                    s.emit_tup_elt(0, || t0.encode(s));
                    s.emit_tup_elt(1, || t1.encode(s));
                    s.emit_tup_elt(2, || t2.encode(s));
                }
            }
        }
    }
}

pub impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>
> (T0, T1, T2): Decodable<D> {
    static fn decode(&self, d: &D) -> (T0, T1, T2) {
        do d.read_tup(3) {
            (
                d.read_tup_elt(0, || decode(d)),
                d.read_tup_elt(1, || decode(d)),
                d.read_tup_elt(2, || decode(d))
            )
        }
    }
}

pub impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>,
    T3: Encodable<S>
> (T0, T1, T2, T3): Encodable<S> {
    fn encode(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3) => {
                do s.emit_tup(4) {
                    s.emit_tup_elt(0, || t0.encode(s));
                    s.emit_tup_elt(1, || t1.encode(s));
                    s.emit_tup_elt(2, || t2.encode(s));
                    s.emit_tup_elt(3, || t3.encode(s));
                }
            }
        }
    }
}

pub impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>,
    T3: Decodable<D>
> (T0, T1, T2, T3): Decodable<D> {
    static fn decode(&self, d: &D) -> (T0, T1, T2, T3) {
        do d.read_tup(4) {
            (
                d.read_tup_elt(0, || decode(d)),
                d.read_tup_elt(1, || decode(d)),
                d.read_tup_elt(2, || decode(d)),
                d.read_tup_elt(3, || decode(d))
            )
        }
    }
}

pub impl<
    S: Encoder,
    T0: Encodable<S>,
    T1: Encodable<S>,
    T2: Encodable<S>,
    T3: Encodable<S>,
    T4: Encodable<S>
> (T0, T1, T2, T3, T4): Encodable<S> {
    fn encode(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3, ref t4) => {
                do s.emit_tup(5) {
                    s.emit_tup_elt(0, || t0.encode(s));
                    s.emit_tup_elt(1, || t1.encode(s));
                    s.emit_tup_elt(2, || t2.encode(s));
                    s.emit_tup_elt(3, || t3.encode(s));
                    s.emit_tup_elt(4, || t4.encode(s));
                }
            }
        }
    }
}

pub impl<
    D: Decoder,
    T0: Decodable<D>,
    T1: Decodable<D>,
    T2: Decodable<D>,
    T3: Decodable<D>,
    T4: Decodable<D>
> (T0, T1, T2, T3, T4): Decodable<D> {
    static fn decode(&self, d: &D)
      -> (T0, T1, T2, T3, T4) {
        do d.read_tup(5) {
            (
                d.read_tup_elt(0, || decode(d)),
                d.read_tup_elt(1, || decode(d)),
                d.read_tup_elt(2, || decode(d)),
                d.read_tup_elt(3, || decode(d)),
                d.read_tup_elt(4, || decode(d))
            )
        }
    }
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

pub trait EncoderHelpers {
    fn emit_from_vec<T>(&self, v: ~[T], f: fn(v: &T));
}

pub impl<S: Encoder> S: EncoderHelpers {
    fn emit_from_vec<T>(&self, v: ~[T], f: fn(v: &T)) {
        do self.emit_owned_vec(v.len()) {
            for v.eachi |i, e| {
                do self.emit_vec_elt(i) {
                    f(e)
                }
            }
        }
    }
}

pub trait DecoderHelpers {
    fn read_to_vec<T>(&self, f: fn() -> T) -> ~[T];
}

pub impl<D: Decoder> D: DecoderHelpers {
    fn read_to_vec<T>(&self, f: fn() -> T) -> ~[T] {
        do self.read_owned_vec |len| {
            do vec::from_fn(len) |i| {
                self.read_vec_elt(i, || f())
            }
        }
    }
}
}

pub use serialize::traits::*;
