/*
Module: serialization

Support code for serialization.
*/

import list::list;
import ebml::writer;

// Set to true to generate more debugging in EBML serialization.
// Totally lame approach.
const debug: bool = true;

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

enum ebml_serializer_tag {
    es_uint, es_u64, es_u32, es_u16, es_u8,
    es_int, es_i64, es_i32, es_i16, es_i8,
    es_bool,
    es_str,
    es_f64, es_f32, es_float,
    es_enum, es_enum_vid, es_enum_body,
    es_vec, es_vec_len, es_vec_elt,

    es_label // Used only when debugging
}

impl of serializer for ebml::writer {
    fn emit_nil() {}

    // used internally to emit things like the vector length and so on
    fn _emit_tagged_uint(t: ebml_serializer_tag, v: uint) {
        assert v <= 0xFFFF_FFFF_u;
        self.wr_tagged_u32(t as uint, v as u32);
    }

    fn _emit_label(label: str) {
        // There are various strings that we have access to, such as
        // the name of a record field, which do not actually appear in
        // the serialized EBML (normally).  This is just for
        // efficiency.  When debugging, though, we can emit such
        // labels and then they will be checked by deserializer to
        // try and check failures more quickly.
        if debug { self.wr_tagged_str(es_label as uint, label) }
    }

    fn emit_uint(v: uint) { self.wr_tagged_u64(es_uint as uint, v as u64); }
    fn emit_u64(v: u64) { self.wr_tagged_u64(es_u64 as uint, v); }
    fn emit_u32(v: u32) { self.wr_tagged_u32(es_u32 as uint, v); }
    fn emit_u16(v: u16) { self.wr_tagged_u16(es_u16 as uint, v); }
    fn emit_u8(v: u8)   { self.wr_tagged_u8 (es_u8  as uint, v); }

    fn emit_int(v: int) { self.wr_tagged_i64(es_int as uint, v as i64); }
    fn emit_i64(v: i64) { self.wr_tagged_i64(es_i64 as uint, v); }
    fn emit_i32(v: i32) { self.wr_tagged_i32(es_i32 as uint, v); }
    fn emit_i16(v: i16) { self.wr_tagged_i16(es_i16 as uint, v); }
    fn emit_i8(v: i8)   { self.wr_tagged_i8 (es_i8  as uint, v); }

    fn emit_bool(v: bool) { self.wr_tagged_u8(es_bool as uint, v as u8) }

    fn emit_f64(_v: f64) { fail "TODO"; }
    fn emit_f32(_v: f32) { fail "TODO"; }
    fn emit_float(_v: float) { fail "TODO"; }

    fn emit_str(v: str) { self.wr_tagged_str(es_str as uint, v) }

    fn emit_enum(name: str, f: fn()) {
        self._emit_label(name);
        self.wr_tag(es_enum as uint, f)
    }
    fn emit_enum_variant(_v_name: str, v_id: uint, _cnt: uint, f: fn()) {
        self._emit_tagged_uint(es_enum_vid, v_id);
        self.wr_tag(es_enum_body as uint, f)
    }
    fn emit_enum_variant_arg(_idx: uint, f: fn()) { f() }

    fn emit_vec(len: uint, f: fn()) {
        self.wr_tag(es_vec as uint) {||
            self._emit_tagged_uint(es_vec_len, len);
            f()
        }
    }

    fn emit_vec_elt(_idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint, f)
    }

    fn emit_box(f: fn()) { f() }
    fn emit_uniq(f: fn()) { f() }
    fn emit_rec(f: fn()) { f() }
    fn emit_rec_field(f_name: str, _f_idx: uint, f: fn()) {
        self._emit_label(f_name);
        f()
    }
    fn emit_tup(_sz: uint, f: fn()) { f() }
    fn emit_tup_elt(_idx: uint, f: fn()) { f() }
}

type ebml_deserializer = {mutable parent: ebml::doc,
                          mutable pos: uint};

fn mk_ebml_deserializer(d: ebml::doc) -> ebml_deserializer {
    {mutable parent: d, mutable pos: d.start}
}

impl of deserializer for ebml_deserializer {
    fn _check_label(lbl: str) {
        if self.pos < self.parent.end {
            let {tag: r_tag, doc: r_doc} =
                ebml::doc_at(self.parent.data, self.pos);
            if r_tag == (es_label as uint) {
                self.pos = r_doc.end;
                let str = ebml::doc_as_str(r_doc);
                if lbl != str {
                    fail #fmt["Expected label %s but found %s", lbl, str];
                }
            }
        }
    }

    fn next_doc(exp_tag: ebml_serializer_tag) -> ebml::doc {
        #debug[". next_doc(exp_tag=%?)", exp_tag];
        if self.pos >= self.parent.end {
            fail "no more documents in current node!";
        }
        let {tag: r_tag, doc: r_doc} =
            ebml::doc_at(self.parent.data, self.pos);
        #debug["self.parent=%?-%? self.pos=%? r_tag=%? r_doc=%?-%?",
               self.parent.start, self.parent.end, self.pos,
               r_tag, r_doc.start, r_doc.end];
        if r_tag != (exp_tag as uint) {
            fail #fmt["expected EMBL doc with tag %? but found tag %?",
                      exp_tag, r_tag];
        }
        if r_doc.end > self.parent.end {
            fail #fmt["invalid EBML, child extends to 0x%x, parent to 0x%x",
                      r_doc.end, self.parent.end];
        }
        self.pos = r_doc.end;
        ret r_doc;
    }

    fn push_doc<T: copy>(d: ebml::doc, f: fn() -> T) -> T{
        let old_parent = self.parent;
        let old_pos = self.pos;
        self.parent = d;
        self.pos = d.start;
        let r = f();
        self.parent = old_parent;
        self.pos = old_pos;
        ret r;
    }

    fn _next_uint(exp_tag: ebml_serializer_tag) -> uint {
        let r = ebml::doc_as_u32(self.next_doc(exp_tag));
        #debug["_next_uint exp_tag=%? result=%?", exp_tag, r];
        ret r as uint;
    }

    fn read_nil() -> () { () }

    fn read_u64() -> u64 { ebml::doc_as_u64(self.next_doc(es_u64)) }
    fn read_u32() -> u32 { ebml::doc_as_u32(self.next_doc(es_u32)) }
    fn read_u16() -> u16 { ebml::doc_as_u16(self.next_doc(es_u16)) }
    fn read_u8 () -> u8  { ebml::doc_as_u8 (self.next_doc(es_u8 )) }
    fn read_uint() -> uint {
        let v = ebml::doc_as_u64(self.next_doc(es_uint));
        if v > (uint::max_value as u64) {
            fail #fmt["uint %? too large for this architecture", v];
        }
        ret v as uint;
    }

    fn read_i64() -> i64 { ebml::doc_as_u64(self.next_doc(es_i64)) as i64 }
    fn read_i32() -> i32 { ebml::doc_as_u32(self.next_doc(es_i32)) as i32 }
    fn read_i16() -> i16 { ebml::doc_as_u16(self.next_doc(es_i16)) as i16 }
    fn read_i8 () -> i8  { ebml::doc_as_u8 (self.next_doc(es_i8 )) as i8  }
    fn read_int() -> int {
        let v = ebml::doc_as_u64(self.next_doc(es_int)) as i64;
        if v > (int::max_value as i64) || v < (int::min_value as i64) {
            fail #fmt["int %? out of range for this architecture", v];
        }
        ret v as int;
    }

    fn read_bool() -> bool { ebml::doc_as_u8(self.next_doc(es_bool)) as bool }

    fn read_f64() -> f64 { fail "read_f64()"; }
    fn read_f32() -> f32 { fail "read_f32()"; }
    fn read_float() -> float { fail "read_float()"; }

    fn read_str() -> str { ebml::doc_as_str(self.next_doc(es_str)) }

    // Compound types:
    fn read_enum<T:copy>(name: str, f: fn() -> T) -> T {
        #debug["read_enum(%s)", name];
        self._check_label(name);
        self.push_doc(self.next_doc(es_enum), f)
    }

    fn read_enum_variant<T:copy>(f: fn(uint) -> T) -> T {
        #debug["read_enum_variant()"];
        let idx = self._next_uint(es_enum_vid);
        #debug["  idx=%u", idx];
        self.push_doc(self.next_doc(es_enum_body)) {||
            f(idx)
        }
    }

    fn read_enum_variant_arg<T:copy>(idx: uint, f: fn() -> T) -> T {
        #debug["read_enum_variant_arg(idx=%u)", idx];
        f()
    }

    fn read_vec<T:copy>(f: fn(uint) -> T) -> T {
        #debug["read_vec()"];
        self.push_doc(self.next_doc(es_vec)) {||
            let len = self._next_uint(es_vec_len);
            #debug["  len=%u", len];
            f(len)
        }
    }

    fn read_vec_elt<T:copy>(idx: uint, f: fn() -> T) -> T {
        #debug["read_vec_elt(idx=%u)", idx];
        self.push_doc(self.next_doc(es_vec_elt), f)
    }

    fn read_box<T:copy>(f: fn() -> T) -> T {
        #debug["read_box()"];
        f()
    }

    fn read_uniq<T:copy>(f: fn() -> T) -> T {
        #debug["read_uniq()"];
        f()
    }

    fn read_rec<T:copy>(f: fn() -> T) -> T {
        #debug["read_rec()"];
        f()
    }

    fn read_rec_field<T:copy>(f_name: str, f_idx: uint, f: fn() -> T) -> T {
        #debug["read_rec_field(%s, idx=%u)", f_name, f_idx];
        self._check_label(f_name);
        f()
    }

    fn read_tup<T:copy>(sz: uint, f: fn() -> T) -> T {
        #debug["read_tup(sz=%u)", sz];
        f()
    }

    fn read_tup_elt<T:copy>(idx: uint, f: fn() -> T) -> T {
        #debug["read_tup_elt(idx=%u)", idx];
        f()
    }
}

// ___________________________________________________________________________
// Helper routines
//
// These should eventually be coded as traits.

impl serializer_helpers<S: serializer> for S {
    fn emit_from_vec<T>(v: [T], f: fn(T)) {
        self.emit_vec(vec::len(v)) {||
            vec::iteri(v) {|i,e|
                self.emit_vec_elt(i) {||
                    f(e)
                }
            }
        }
    }
}

impl deserializer_helpers<D: deserializer> for D {
    fn read_to_vec<T>(f: fn() -> T) -> [T] {
        self.read_vec {|len|
            let v = [];
            vec::reserve(v, len);
            uint::range(0u, len) {|i|
                self.read_vec_elt(i) {|| v += [f()] }
            }
            v
        }
    }
}

// ___________________________________________________________________________
// Testing

#[test]
fn test_option_int() {
    fn serialize_1<S: serializer>(s: S, v: int) {
        s.emit_i64(v as i64);
    }

    fn serialize_0<S: serializer>(s: S, v: option<int>) {
        s.emit_enum("core::option::t") {||
            alt v {
              none {
                s.emit_enum_variant("core::option::none", 0u, 0u) {||}
              }
              some(v0) {
                s.emit_enum_variant("core::option::some", 1u, 1u) {||
                    s.emit_enum_variant_arg(0u) {|| serialize_1(s, v0) }
                }
              }
            }
        }
    }

    fn deserialize_1<S: deserializer>(s: S) -> int {
        s.read_i64() as int
    }

    fn deserialize_0<S: deserializer>(s: S) -> option<int> {
        s.read_enum("option") {||
            s.read_enum_variant {|i|
                alt check i {
                  0u { none }
                  1u {
                    let v0 = s.read_enum_variant_arg(0u) {||
                        deserialize_1(s)
                    };
                    some(v0)
                  }
                }
            }
        }
    }

    fn test_v(v: option<int>) {
        #debug["v == %?", v];
        let mbuf = io::mk_mem_buffer();
        let ebml_w = ebml::mk_writer(io::mem_buffer_writer(mbuf));
        serialize_0(ebml_w, v);
        let ebml_doc = ebml::new_doc(@io::mem_buffer_buf(mbuf));
        let deser = mk_ebml_deserializer(ebml_doc);
        let v1 = deserialize_0(deser);
        #debug["v1 == %?", v1];
        assert v == v1;
    }

    test_v(some(22));
    test_v(none);
    test_v(some(3));
}
