//! Support code for serialization.

/*
Core serialization interfaces.
*/

trait Serializer {
    // Primitive types:
    fn emit_nil();
    fn emit_uint(v: uint);
    fn emit_u64(v: u64);
    fn emit_u32(v: u32);
    fn emit_u16(v: u16);
    fn emit_u8(v: u8);
    fn emit_int(v: int);
    fn emit_i64(v: i64);
    fn emit_i32(v: i32);
    fn emit_i16(v: i16);
    fn emit_i8(v: i8);
    fn emit_bool(v: bool);
    fn emit_float(v: float);
    fn emit_f64(v: f64);
    fn emit_f32(v: f32);
    fn emit_str(v: &str);

    // Compound types:
    fn emit_enum(name: &str, f: fn());
    fn emit_enum_variant(v_name: &str, v_id: uint, sz: uint, f: fn());
    fn emit_enum_variant_arg(idx: uint, f: fn());
    fn emit_vec(len: uint, f: fn());
    fn emit_vec_elt(idx: uint, f: fn());
    fn emit_box(f: fn());
    fn emit_uniq(f: fn());
    fn emit_rec(f: fn());
    fn emit_rec_field(f_name: &str, f_idx: uint, f: fn());
    fn emit_tup(sz: uint, f: fn());
    fn emit_tup_elt(idx: uint, f: fn());
}

trait Deserializer {
    // Primitive types:
    fn read_nil() -> ();
    fn read_uint() -> uint;
    fn read_u64() -> u64;
    fn read_u32() -> u32;
    fn read_u16() -> u16;
    fn read_u8() -> u8;
    fn read_int() -> int;
    fn read_i64() -> i64;
    fn read_i32() -> i32;
    fn read_i16() -> i16;
    fn read_i8() -> i8;
    fn read_bool() -> bool;
    fn read_f64() -> f64;
    fn read_f32() -> f32;
    fn read_float() -> float;
    fn read_str() -> ~str;

    // Compound types:
    fn read_enum<T>(name: ~str, f: fn() -> T) -> T;
    fn read_enum_variant<T>(f: fn(uint) -> T) -> T;
    fn read_enum_variant_arg<T>(idx: uint, f: fn() -> T) -> T;
    fn read_vec<T>(f: fn(uint) -> T) -> T;
    fn read_vec_elt<T>(idx: uint, f: fn() -> T) -> T;
    fn read_box<T>(f: fn() -> T) -> T;
    fn read_uniq<T>(f: fn() -> T) -> T;
    fn read_rec<T>(f: fn() -> T) -> T;
    fn read_rec_field<T>(f_name: ~str, f_idx: uint, f: fn() -> T) -> T;
    fn read_tup<T>(sz: uint, f: fn() -> T) -> T;
    fn read_tup_elt<T>(idx: uint, f: fn() -> T) -> T;
}

trait Serializable {
    fn serialize<S: Serializer>(s: S);
    static fn deserialize<D: Deserializer>(d: D) -> self;
}

impl uint: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_uint(self) }
    static fn deserialize<D: Deserializer>(d: D) -> uint { d.read_uint() }
}

impl u8: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_u8(self) }
    static fn deserialize<D: Deserializer>(d: D) -> u8 { d.read_u8() }
}

impl u16: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_u16(self) }
    static fn deserialize<D: Deserializer>(d: D) -> u16 { d.read_u16() }
}

impl u32: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_u32(self) }
    static fn deserialize<D: Deserializer>(d: D) -> u32 { d.read_u32() }
}

impl u64: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_u64(self) }
    static fn deserialize<D: Deserializer>(d: D) -> u64 { d.read_u64() }
}

impl int: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_int(self) }
    static fn deserialize<D: Deserializer>(d: D) -> int { d.read_int() }
}

impl i8: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_i8(self) }
    static fn deserialize<D: Deserializer>(d: D) -> i8 { d.read_i8() }
}

impl i16: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_i16(self) }
    static fn deserialize<D: Deserializer>(d: D) -> i16 { d.read_i16() }
}

impl i32: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_i32(self) }
    static fn deserialize<D: Deserializer>(d: D) -> i32 { d.read_i32() }
}

impl i64: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_i64(self) }
    static fn deserialize<D: Deserializer>(d: D) -> i64 { d.read_i64() }
}

impl ~str: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_str(self) }
    static fn deserialize<D: Deserializer>(d: D) -> ~str { d.read_str() }
}

impl float: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_float(self) }
    static fn deserialize<D: Deserializer>(d: D) -> float { d.read_float() }
}

impl f32: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_f32(self) }
    static fn deserialize<D: Deserializer>(d: D) -> f32 { d.read_f32() }
}

impl f64: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_f64(self) }
    static fn deserialize<D: Deserializer>(d: D) -> f64 { d.read_f64() }
}

impl bool: Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_bool(self) }
    static fn deserialize<D: Deserializer>(d: D) -> bool { d.read_bool() }
}

impl (): Serializable {
    fn serialize<S: Serializer>(s: S) { s.emit_nil() }
    static fn deserialize<D: Deserializer>(d: D) -> () { d.read_nil() }
}

impl<T: Serializable> @T: Serializable {
    fn serialize<S: Serializer>(s: S) {
        s.emit_box(|| (*self).serialize(s))
    }

    static fn deserialize<D: Deserializer>(d: D) -> @T {
        d.read_box(|| @deserialize(d))
    }
}

impl<T: Serializable> ~T: Serializable {
    fn serialize<S: Serializer>(s: S) {
        s.emit_uniq(|| (*self).serialize(s))
    }

    static fn deserialize<D: Deserializer>(d: D) -> ~T {
        d.read_uniq(|| ~deserialize(d))
    }
}

impl<T: Serializable> ~[T]: Serializable {
    fn serialize<S: Serializer>(s: S) {
        do s.emit_vec(self.len()) {
            for self.eachi |i, e| {
                s.emit_vec_elt(i, || e.serialize(s))
            }
        }
    }

    static fn deserialize<D: Deserializer>(d: D) -> ~[T] {
        do d.read_vec |len| {
            do vec::from_fn(len) |i| {
                d.read_vec_elt(i, || deserialize(d))
            }
        }
    }
}

impl<T: Serializable> Option<T>: Serializable {
    fn serialize<S: Serializer>(s: S) {
        do s.emit_enum(~"option") {
            match self {
              None => do s.emit_enum_variant(~"none", 0u, 0u) {
              },

              Some(v) => do s.emit_enum_variant(~"some", 1u, 1u) {
                s.emit_enum_variant_arg(0u, || v.serialize(s))
              }
            }
        }
    }

    static fn deserialize<D: Deserializer>(d: D) -> Option<T> {
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

impl<
    T0: Serializable,
    T1: Serializable
> (T0, T1): Serializable {
    fn serialize<S: Serializer>(s: S) {
        match self {
            (t0, t1) => {
                do s.emit_tup(2) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                }
            }
        }
    }

    static fn deserialize<D: Deserializer>(d: D) -> (T0, T1) {
        do d.read_tup(2) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d))
            )
        }
    }
}

impl<
    T0: Serializable,
    T1: Serializable,
    T2: Serializable
> (T0, T1, T2): Serializable {
    fn serialize<S: Serializer>(s: S) {
        match self {
            (t0, t1, t2) => {
                do s.emit_tup(3) {
                    s.emit_tup_elt(0, || t0.serialize(s));
                    s.emit_tup_elt(1, || t1.serialize(s));
                    s.emit_tup_elt(2, || t2.serialize(s));
                }
            }
        }
    }

    static fn deserialize<D: Deserializer>(d: D) -> (T0, T1, T2) {
        do d.read_tup(3) {
            (
                d.read_tup_elt(0, || deserialize(d)),
                d.read_tup_elt(1, || deserialize(d)),
                d.read_tup_elt(2, || deserialize(d))
            )
        }
    }
}

impl<
    T0: Serializable,
    T1: Serializable,
    T2: Serializable,
    T3: Serializable
> (T0, T1, T2, T3): Serializable {
    fn serialize<S: Serializer>(s: S) {
        match self {
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

    static fn deserialize<D: Deserializer>(d: D) -> (T0, T1, T2, T3) {
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

impl<
    T0: Serializable,
    T1: Serializable,
    T2: Serializable,
    T3: Serializable,
    T4: Serializable
> (T0, T1, T2, T3, T4): Serializable {
    fn serialize<S: Serializer>(s: S) {
        match self {
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

    static fn deserialize<D: Deserializer>(d: D) -> (T0, T1, T2, T3, T4) {
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

fn emit_from_vec<S: Serializer, T>(s: S, v: ~[T], f: fn(T)) {
    do s.emit_vec(v.len()) {
        for v.eachi |i, e| {
            do s.emit_vec_elt(i) {
                f(*e)
            }
        }
    }
}

fn read_to_vec<D: Deserializer, T: Copy>(d: D, f: fn() -> T) -> ~[T] {
    do d.read_vec |len| {
        do vec::from_fn(len) |i| {
            d.read_vec_elt(i, || f())
        }
    }
}

trait SerializerHelpers {
    fn emit_from_vec<T>(v: ~[T], f: fn(T));
}

impl<S: Serializer> S: SerializerHelpers {
    fn emit_from_vec<T>(v: ~[T], f: fn(T)) {
        emit_from_vec(self, v, f)
    }
}

trait DeserializerHelpers {
    fn read_to_vec<T: Copy>(f: fn() -> T) -> ~[T];
}

impl<D: Deserializer> D: DeserializerHelpers {
    fn read_to_vec<T: Copy>(f: fn() -> T) -> ~[T] {
        read_to_vec(self, f)
    }
}
