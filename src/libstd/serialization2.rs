//! Support code for serialization.

/*
Core serialization interfaces.
*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
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
    fn emit_str(&self, v: &str);

    // Compound types:
    fn emit_enum(&self, name: &str, f: fn());
    fn emit_enum_variant(&self, v_name: &str, v_id: uint, sz: uint, f: fn());
    fn emit_enum_variant_arg(&self, idx: uint, f: fn());
    fn emit_vec(&self, len: uint, f: fn());
    fn emit_vec_elt(&self, idx: uint, f: fn());
    fn emit_box(&self, f: fn());
    fn emit_uniq(&self, f: fn());
    fn emit_rec(&self, f: fn());
    fn emit_rec_field(&self, f_name: &str, f_idx: uint, f: fn());
    fn emit_tup(&self, sz: uint, f: fn());
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
    fn read_str(&self) -> ~str;

    // Compound types:
    fn read_enum<T>(&self, name: ~str, f: fn() -> T) -> T;
    fn read_enum_variant<T>(&self, f: fn(uint) -> T) -> T;
    fn read_enum_variant_arg<T>(&self, idx: uint, f: fn() -> T) -> T;
    fn read_vec<T>(&self, f: fn(uint) -> T) -> T;
    fn read_vec_elt<T>(&self, idx: uint, f: fn() -> T) -> T;
    fn read_box<T>(&self, f: fn() -> T) -> T;
    fn read_uniq<T>(&self, f: fn() -> T) -> T;
    fn read_rec<T>(&self, f: fn() -> T) -> T;
    fn read_rec_field<T>(&self, f_name: ~str, f_idx: uint, f: fn() -> T) -> T;
    fn read_tup<T>(&self, sz: uint, f: fn() -> T) -> T;
    fn read_tup_elt<T>(&self, idx: uint, f: fn() -> T) -> T;
}

pub trait Serializable {
    fn serialize<S: Serializer>(&self, s: &S);
    static fn deserialize<D: Deserializer>(&self, d: &D) -> self;
}

pub impl uint: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_uint(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> uint {
        d.read_uint()
    }
}

pub impl u8: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_u8(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> u8 {
        d.read_u8()
    }
}

pub impl u16: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_u16(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> u16 {
        d.read_u16()
    }
}

pub impl u32: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_u32(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> u32 {
        d.read_u32()
    }
}

pub impl u64: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_u64(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> u64 {
        d.read_u64()
    }
}

pub impl int: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_int(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> int {
        d.read_int()
    }
}

pub impl i8: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_i8(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> i8 {
        d.read_i8()
    }
}

pub impl i16: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_i16(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> i16 {
        d.read_i16()
    }
}

pub impl i32: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_i32(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> i32 {
        d.read_i32()
    }
}

pub impl i64: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_i64(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> i64 {
        d.read_i64()
    }
}

pub impl ~str: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_str(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> ~str {
        d.read_str()
    }
}

pub impl float: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_float(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> float {
        d.read_float()
    }
}

pub impl f32: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_f32(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> f32 {
        d.read_f32() }
}

pub impl f64: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_f64(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> f64 {
        d.read_f64()
    }
}

pub impl bool: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_bool(*self) }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> bool {
        d.read_bool()
    }
}

pub impl (): Serializable {
    fn serialize<S: Serializer>(&self, s: &S) { s.emit_nil() }
    static fn deserialize<D: Deserializer>(&self, d: &D) -> () {
        d.read_nil()
    }
}

pub impl<T: Serializable> @T: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        s.emit_box(|| (*self).serialize(s))
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> @T {
        d.read_box(|| @deserialize(d))
    }
}

pub impl<T: Serializable> ~T: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        s.emit_uniq(|| (*self).serialize(s))
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> ~T {
        d.read_uniq(|| ~deserialize(d))
    }
}

pub impl<T: Serializable> ~[T]: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        do s.emit_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.serialize(s))
            }
        }
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> ~[T] {
        do d.read_vec |len| {
            do vec::from_fn(len) |i| {
                d.read_vec_elt(i, || deserialize(d))
            }
        }
    }
}

pub impl<T: Serializable> Option<T>: Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        do s.emit_enum(~"option") {
            match *self {
              None => do s.emit_enum_variant(~"none", 0u, 0u) {
              },

              Some(v) => do s.emit_enum_variant(~"some", 1u, 1u) {
                s.emit_enum_variant_arg(0u, || v.serialize(s))
              }
            }
        }
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> Option<T> {
        do d.read_enum(~"option") {
            do d.read_enum_variant |i| {
                match i {
                  0 => None,
                  1 => Some(d.read_enum_variant_arg(0u, || deserialize(d))),
                  _ => fail(#fmt("Bad variant for option: %u", i))
                }
            }
        }
    }
}

pub impl<
    T0: Serializable,
    T1: Serializable
> (T0, T1): Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        match *self {
            (t0, t1) => {
                do s.emit_tup(2) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                }
            }
        }
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> (T0, T1) {
        do d.read_tup(2) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d))
            )
        }
    }
}

pub impl<
    T0: Serializable,
    T1: Serializable,
    T2: Serializable
> (T0, T1, T2): Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        match *self {
            (t0, t1, t2) => {
                do s.emit_tup(3) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                }
            }
        }
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> (T0, T1, T2) {
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
    T0: Serializable,
    T1: Serializable,
    T2: Serializable,
    T3: Serializable
> (T0, T1, T2, T3): Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        match *self {
            (t0, t1, t2, t3) => {
                do s.emit_tup(4) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                    s.emit_tup_elt(3, || t3.serialize(s));
                }
            }
        }
    }

    static fn deserialize<D: Deserializer>(&self, d: &D) -> (T0, T1, T2, T3) {
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
    T0: Serializable,
    T1: Serializable,
    T2: Serializable,
    T3: Serializable,
    T4: Serializable
> (T0, T1, T2, T3, T4): Serializable {
    fn serialize<S: Serializer>(&self, s: &S) {
        match *self {
            (t0, t1, t2, t3, t4) => {
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

    static fn deserialize<D: Deserializer>(&self, d: &D)
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
        do self.emit_vec(v.len()) {
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
        do self.read_vec |len| {
            do vec::from_fn(len) |i| {
                self.read_vec_elt(i, || f())
            }
        }
    }
}
