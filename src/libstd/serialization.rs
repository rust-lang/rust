//! Support code for serialization.

/*
Core serialization interfaces.
*/

#[forbid(deprecated_mode)];
#[forbid(non_camel_case_types)];

pub trait Serializer {
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

pub trait Deserializer {
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
pub trait Serializable<S: Serializer> {
    fn serialize(&self, s: &S);
}

pub trait Deserializable<D: Deserializer> {
    static fn deserialize(&self, d: &D) -> self;
}

pub impl<S: Serializer> uint: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_uint(*self) }
}

pub impl<D: Deserializer> uint: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> uint {
        d.read_uint()
    }
}

pub impl<S: Serializer> u8: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_u8(*self) }
}

pub impl<D: Deserializer> u8: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> u8 {
        d.read_u8()
    }
}

pub impl<S: Serializer> u16: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_u16(*self) }
}

pub impl<D: Deserializer> u16: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> u16 {
        d.read_u16()
    }
}

pub impl<S: Serializer> u32: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_u32(*self) }
}

pub impl<D: Deserializer> u32: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> u32 {
        d.read_u32()
    }
}

pub impl<S: Serializer> u64: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_u64(*self) }
}

pub impl<D: Deserializer> u64: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> u64 {
        d.read_u64()
    }
}

pub impl<S: Serializer> int: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_int(*self) }
}

pub impl<D: Deserializer> int: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> int {
        d.read_int()
    }
}

pub impl<S: Serializer> i8: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_i8(*self) }
}

pub impl<D: Deserializer> i8: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> i8 {
        d.read_i8()
    }
}

pub impl<S: Serializer> i16: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_i16(*self) }
}

pub impl<D: Deserializer> i16: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> i16 {
        d.read_i16()
    }
}

pub impl<S: Serializer> i32: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_i32(*self) }
}

pub impl<D: Deserializer> i32: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> i32 {
        d.read_i32()
    }
}

pub impl<S: Serializer> i64: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_i64(*self) }
}

pub impl<D: Deserializer> i64: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> i64 {
        d.read_i64()
    }
}

pub impl<S: Serializer> &str: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_borrowed_str(*self) }
}

pub impl<S: Serializer> ~str: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_owned_str(*self) }
}

pub impl<D: Deserializer> ~str: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> ~str {
        d.read_owned_str()
    }
}

pub impl<S: Serializer> @str: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_managed_str(*self) }
}

pub impl<D: Deserializer> @str: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> @str {
        d.read_managed_str()
    }
}

pub impl<S: Serializer> float: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_float(*self) }
}

pub impl<D: Deserializer> float: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> float {
        d.read_float()
    }
}

pub impl<S: Serializer> f32: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_f32(*self) }
}

pub impl<D: Deserializer> f32: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> f32 {
        d.read_f32() }
}

pub impl<S: Serializer> f64: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_f64(*self) }
}

pub impl<D: Deserializer> f64: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> f64 {
        d.read_f64()
    }
}

pub impl<S: Serializer> bool: Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_bool(*self) }
}

pub impl<D: Deserializer> bool: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> bool {
        d.read_bool()
    }
}

pub impl<S: Serializer> (): Serializable<S> {
    fn serialize(&self, s: &S) { s.emit_nil() }
}

pub impl<D: Deserializer> (): Deserializable<D> {
    static fn deserialize(&self, d: &D) -> () {
        d.read_nil()
    }
}

pub impl<S: Serializer, T: Serializable<S>> &T: Serializable<S> {
    fn serialize(&self, s: &S) {
        s.emit_borrowed(|| (**self).serialize(s))
    }
}

pub impl<S: Serializer, T: Serializable<S>> ~T: Serializable<S> {
    fn serialize(&self, s: &S) {
        s.emit_owned(|| (**self).serialize(s))
    }
}

pub impl<D: Deserializer, T: Deserializable<D>> ~T: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> ~T {
        d.read_owned(|| ~deserialize(d))
    }
}

pub impl<S: Serializer, T: Serializable<S>> @T: Serializable<S> {
    fn serialize(&self, s: &S) {
        s.emit_managed(|| (**self).serialize(s))
    }
}

pub impl<D: Deserializer, T: Deserializable<D>> @T: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> @T {
        d.read_managed(|| @deserialize(d))
    }
}

pub impl<S: Serializer, T: Serializable<S>> &[T]: Serializable<S> {
    fn serialize(&self, s: &S) {
        do s.emit_borrowed_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.serialize(s))
            }
        }
    }
}

pub impl<S: Serializer, T: Serializable<S>> ~[T]: Serializable<S> {
    fn serialize(&self, s: &S) {
        do s.emit_owned_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.serialize(s))
            }
        }
    }
}

pub impl<D: Deserializer, T: Deserializable<D>> ~[T]: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> ~[T] {
        do d.read_owned_vec |len| {
            do vec::from_fn(len) |i| {
                d.read_vec_elt(i, || deserialize(d))
            }
        }
    }
}

pub impl<S: Serializer, T: Serializable<S>> @[T]: Serializable<S> {
    fn serialize(&self, s: &S) {
        do s.emit_managed_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.serialize(s))
            }
        }
    }
}

pub impl<D: Deserializer, T: Deserializable<D>> @[T]: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> @[T] {
        do d.read_managed_vec |len| {
            do at_vec::from_fn(len) |i| {
                d.read_vec_elt(i, || deserialize(d))
            }
        }
    }
}

pub impl<S: Serializer, T: Serializable<S>> Option<T>: Serializable<S> {
    fn serialize(&self, s: &S) {
        do s.emit_enum(~"option") {
            match *self {
              None => do s.emit_enum_variant(~"none", 0u, 0u) {
              },

              Some(ref v) => do s.emit_enum_variant(~"some", 1u, 1u) {
                s.emit_enum_variant_arg(0u, || v.serialize(s))
              }
            }
        }
    }
}

pub impl<D: Deserializer, T: Deserializable<D>> Option<T>: Deserializable<D> {
    static fn deserialize(&self, d: &D) -> Option<T> {
        do d.read_enum(~"option") {
            do d.read_enum_variant |i| {
                match i {
                  0 => None,
                  1 => Some(d.read_enum_variant_arg(0u, || deserialize(d))),
                  _ => fail(fmt!("Bad variant for option: %u", i))
                }
            }
        }
    }
}

pub impl<
    S: Serializer,
    T0: Serializable<S>,
    T1: Serializable<S>
> (T0, T1): Serializable<S> {
    fn serialize(&self, s: &S) {
        match *self {
            (ref t0, ref t1) => {
                do s.emit_tup(2) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                }
            }
        }
    }
}

pub impl<
    D: Deserializer,
    T0: Deserializable<D>,
    T1: Deserializable<D>
> (T0, T1): Deserializable<D> {
    static fn deserialize(&self, d: &D) -> (T0, T1) {
        do d.read_tup(2) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d))
            )
        }
    }
}

pub impl<
    S: Serializer,
    T0: Serializable<S>,
    T1: Serializable<S>,
    T2: Serializable<S>
> (T0, T1, T2): Serializable<S> {
    fn serialize(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2) => {
                do s.emit_tup(3) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                }
            }
        }
    }
}

pub impl<
    D: Deserializer,
    T0: Deserializable<D>,
    T1: Deserializable<D>,
    T2: Deserializable<D>
> (T0, T1, T2): Deserializable<D> {
    static fn deserialize(&self, d: &D) -> (T0, T1, T2) {
        do d.read_tup(3) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d)),
                d.read_tup_elt(2, || deserialize(d))
            )
        }
    }
}

pub impl<
    S: Serializer,
    T0: Serializable<S>,
    T1: Serializable<S>,
    T2: Serializable<S>,
    T3: Serializable<S>
> (T0, T1, T2, T3): Serializable<S> {
    fn serialize(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3) => {
                do s.emit_tup(4) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                    s.emit_tup_elt(3, || t3.serialize(s));
                }
            }
        }
    }
}

pub impl<
    D: Deserializer,
    T0: Deserializable<D>,
    T1: Deserializable<D>,
    T2: Deserializable<D>,
    T3: Deserializable<D>
> (T0, T1, T2, T3): Deserializable<D> {
    static fn deserialize(&self, d: &D) -> (T0, T1, T2, T3) {
        do d.read_tup(4) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d)),
                d.read_tup_elt(2, || deserialize(d)),
                d.read_tup_elt(3, || deserialize(d))
            )
        }
    }
}

pub impl<
    S: Serializer,
    T0: Serializable<S>,
    T1: Serializable<S>,
    T2: Serializable<S>,
    T3: Serializable<S>,
    T4: Serializable<S>
> (T0, T1, T2, T3, T4): Serializable<S> {
    fn serialize(&self, s: &S) {
        match *self {
            (ref t0, ref t1, ref t2, ref t3, ref t4) => {
                do s.emit_tup(5) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                    s.emit_tup_elt(3, || t3.serialize(s));
                    s.emit_tup_elt(4, || t4.serialize(s));
                }
            }
        }
    }
}

pub impl<
    D: Deserializer,
    T0: Deserializable<D>,
    T1: Deserializable<D>,
    T2: Deserializable<D>,
    T3: Deserializable<D>,
    T4: Deserializable<D>
> (T0, T1, T2, T3, T4): Deserializable<D> {
    static fn deserialize(&self, d: &D)
      -> (T0, T1, T2, T3, T4) {
        do d.read_tup(5) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d)),
                d.read_tup_elt(2, || deserialize(d)),
                d.read_tup_elt(3, || deserialize(d)),
                d.read_tup_elt(4, || deserialize(d))
            )
        }
    }
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

pub trait SerializerHelpers {
    fn emit_from_vec<T>(&self, v: ~[T], f: fn(v: &T));
}

pub impl<S: Serializer> S: SerializerHelpers {
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

pub trait DeserializerHelpers {
    fn read_to_vec<T>(&self, f: fn() -> T) -> ~[T];
}

pub impl<D: Deserializer> D: DeserializerHelpers {
    fn read_to_vec<T>(&self, f: fn() -> T) -> ~[T] {
        do self.read_owned_vec |len| {
            do vec::from_fn(len) |i| {
                self.read_vec_elt(i, || f())
            }
        }
    }
}
}

pub use traits::*;
