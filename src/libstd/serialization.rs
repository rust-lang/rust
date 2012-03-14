#[doc = "Support code for serialization."];

use core;

/*
Core serialization interfaces.
*/

iface serializer {
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
    fn emit_str(v: str);

    // Compound types:
    fn emit_enum(name: str, f: fn());
    fn emit_enum_variant(v_name: str, v_id: uint, sz: uint, f: fn());
    fn emit_enum_variant_arg(idx: uint, f: fn());
    fn emit_vec(len: uint, f: fn());
    fn emit_vec_elt(idx: uint, f: fn());
    fn emit_box(f: fn());
    fn emit_uniq(f: fn());
    fn emit_rec(f: fn());
    fn emit_rec_field(f_name: str, f_idx: uint, f: fn());
    fn emit_tup(sz: uint, f: fn());
    fn emit_tup_elt(idx: uint, f: fn());
}

iface deserializer {
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

    fn read_str() -> str;

    fn read_f64() -> f64;
    fn read_f32() -> f32;
    fn read_float() -> float;

    // Compound types:
    fn read_enum<T:copy>(name: str, f: fn() -> T) -> T;
    fn read_enum_variant<T:copy>(f: fn(uint) -> T) -> T;
    fn read_enum_variant_arg<T:copy>(idx: uint, f: fn() -> T) -> T;
    fn read_vec<T:copy>(f: fn(uint) -> T) -> T;
    fn read_vec_elt<T:copy>(idx: uint, f: fn() -> T) -> T;
    fn read_box<T:copy>(f: fn() -> T) -> T;
    fn read_uniq<T:copy>(f: fn() -> T) -> T;
    fn read_rec<T:copy>(f: fn() -> T) -> T;
    fn read_rec_field<T:copy>(f_name: str, f_idx: uint, f: fn() -> T) -> T;
    fn read_tup<T:copy>(sz: uint, f: fn() -> T) -> T;
    fn read_tup_elt<T:copy>(idx: uint, f: fn() -> T) -> T;
}

// ___________________________________________________________________________
// Helper routines
//
// In some cases, these should eventually be coded as traits.

fn emit_from_vec<S: serializer, T>(s: S, v: [T], f: fn(T)) {
    s.emit_vec(vec::len(v)) {||
        vec::iteri(v) {|i,e|
            s.emit_vec_elt(i) {||
                f(e)
            }
        }
    }
}

fn read_to_vec<D: deserializer, T>(d: D, f: fn() -> T) -> [T] {
    d.read_vec {|len|
        vec::from_fn(len) {|i|
            d.read_vec_elt(i) {|| f() }
        }
    }
}

impl serializer_helpers<S: serializer> for S {
    fn emit_from_vec<T>(v: [T], f: fn(T)) {
        emit_from_vec(self, v, f)
    }
}

impl deserializer_helpers<D: deserializer> for D {
    fn read_to_vec<T>(f: fn() -> T) -> [T] {
        read_to_vec(self, f)
    }
}

fn serialize_uint<S: serializer>(s: S, v: uint) {
    s.emit_uint(v);
}

fn deserialize_uint<D: deserializer>(d: D) -> uint {
    d.read_uint()
}
