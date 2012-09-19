//! Support code for serialization.

// XXX remove
#[cfg(stage0)]
#[allow(non_camel_case_types)]
type serializer = Serializer;
#[cfg(stage0)]
#[allow(non_camel_case_types)]
type deserializer = Deserializer;

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

    fn read_str() -> ~str;

    fn read_f64() -> f64;
    fn read_f32() -> f32;
    fn read_float() -> float;

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

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

fn emit_from_vec<S: Serializer, T>(s: S, v: ~[T], f: fn(T)) {
    do s.emit_vec(vec::len(v)) {
        do vec::iteri(v) |i,e| {
            do s.emit_vec_elt(i) {
                f(e)
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

fn serialize_uint<S: Serializer>(s: S, v: uint) {
    s.emit_uint(v);
}

fn deserialize_uint<D: Deserializer>(d: D) -> uint {
    d.read_uint()
}

fn serialize_u8<S: Serializer>(s: S, v: u8) {
    s.emit_u8(v);
}

fn deserialize_u8<D: Deserializer>(d: D) -> u8 {
    d.read_u8()
}

fn serialize_u16<S: Serializer>(s: S, v: u16) {
    s.emit_u16(v);
}

fn deserialize_u16<D: Deserializer>(d: D) -> u16 {
    d.read_u16()
}

fn serialize_u32<S: Serializer>(s: S, v: u32) {
    s.emit_u32(v);
}

fn deserialize_u32<D: Deserializer>(d: D) -> u32 {
    d.read_u32()
}

fn serialize_u64<S: Serializer>(s: S, v: u64) {
    s.emit_u64(v);
}

fn deserialize_u64<D: Deserializer>(d: D) -> u64 {
    d.read_u64()
}

fn serialize_int<S: Serializer>(s: S, v: int) {
    s.emit_int(v);
}

fn deserialize_int<D: Deserializer>(d: D) -> int {
    d.read_int()
}

fn serialize_i8<S: Serializer>(s: S, v: i8) {
    s.emit_i8(v);
}

fn deserialize_i8<D: Deserializer>(d: D) -> i8 {
    d.read_i8()
}

fn serialize_i16<S: Serializer>(s: S, v: i16) {
    s.emit_i16(v);
}

fn deserialize_i16<D: Deserializer>(d: D) -> i16 {
    d.read_i16()
}

fn serialize_i32<S: Serializer>(s: S, v: i32) {
    s.emit_i32(v);
}

fn deserialize_i32<D: Deserializer>(d: D) -> i32 {
    d.read_i32()
}

fn serialize_i64<S: Serializer>(s: S, v: i64) {
    s.emit_i64(v);
}

fn deserialize_i64<D: Deserializer>(d: D) -> i64 {
    d.read_i64()
}

fn serialize_str<S: Serializer>(s: S, v: &str) {
    s.emit_str(v);
}

fn deserialize_str<D: Deserializer>(d: D) -> ~str {
    d.read_str()
}

fn serialize_float<S: Serializer>(s: S, v: float) {
    s.emit_float(v);
}

fn deserialize_float<D: Deserializer>(d: D) -> float {
    d.read_float()
}

fn serialize_f32<S: Serializer>(s: S, v: f32) {
    s.emit_f32(v);
}

fn deserialize_f32<D: Deserializer>(d: D) -> f32 {
    d.read_f32()
}

fn serialize_f64<S: Serializer>(s: S, v: f64) {
    s.emit_f64(v);
}

fn deserialize_f64<D: Deserializer>(d: D) -> f64 {
    d.read_f64()
}

fn serialize_bool<S: Serializer>(s: S, v: bool) {
    s.emit_bool(v);
}

fn deserialize_bool<D: Deserializer>(d: D) -> bool {
    d.read_bool()
}

fn serialize_Option<S: Serializer,T>(s: S, v: Option<T>, st: fn(T)) {
    do s.emit_enum(~"option") {
        match v {
          None => do s.emit_enum_variant(~"none", 0u, 0u) {
          },

          Some(v) => do s.emit_enum_variant(~"some", 1u, 1u) {
            do s.emit_enum_variant_arg(0u) {
                st(v)
            }
          }
        }
    }
}

fn deserialize_Option<D: Deserializer,T: Copy>(d: D, st: fn() -> T)
    -> Option<T> {
    do d.read_enum(~"option") {
        do d.read_enum_variant |i| {
            match i {
              0 => None,
              1 => Some(d.read_enum_variant_arg(0u, || st() )),
              _ => fail(#fmt("Bad variant for option: %u", i))
            }
        }
    }
}
