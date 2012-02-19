/*
Module: serialization

Support code for serialization.
*/

import list::list;
import ebml::writer_util;

iface serializer {
    // Primitive types:
    fn emit_nil();
    fn emit_u64(v: u64);
    fn emit_i64(v: i64);
    fn emit_bool(v: bool);
    fn emit_f64(v: f64);
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
    fn read_u64() -> u64;
    fn read_i64() -> i64;
    fn read_bool() -> bool;
    fn read_f64() -> f64;
    fn read_str() -> str;

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
    es_u64, es_i64, es_bool,
    es_str,
    es_enum, es_enum_vid, es_enum_body,
    es_vec, es_vec_len, es_vec_elt
}

impl of serializer for ebml::writer {
    fn emit_nil() {}

    fn emit_num(tag: ebml_serializer_tag, v: u64) {
        self.wr_tag(tag as uint) {|| self.wr_vu64(v) }
    }

    fn emit_u64(v: u64) { self.emit_num(es_u64, v) }
    fn emit_i64(v: i64) { self.emit_num(es_i64, v as u64) }
    fn emit_bool(v: bool) { self.emit_num(es_bool, v as u64) }
    fn emit_f64(_v: f64) { fail "TODO"; }
    fn emit_str(v: str) { self.wr_tag(es_str as uint) {|| self.wr_str(v) } }

    fn emit_enum(_name: str, f: fn()) {
        self.wr_tag(es_enum as uint) {|| f() }
    }
    fn emit_enum_variant(_v_name: str, v_id: uint, _cnt: uint, f: fn()) {
        self.emit_num(es_enum_vid, v_id as u64);
        self.wr_tag(es_enum_body as uint) {|| f() }
    }
    fn emit_enum_variant_arg(_idx: uint, f: fn()) { f() }

    fn emit_vec(len: uint, f: fn()) {
        self.wr_tag(es_vec as uint) {||
            self.emit_num(es_vec_len, len as u64);
            f()
        }
    }

    fn emit_vec_elt(_idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_vec_elt(_idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_box(f: fn()) { f() }
    fn emit_uniq(f: fn()) { f() }
    fn emit_rec(f: fn()) { f() }
    fn emit_rec_field(_f_name: str, _f_idx: uint, f: fn()) { f() }
    fn emit_tup(_sz: uint, f: fn()) { f() }
    fn emit_tup_elt(_idx: uint, f: fn()) { f() }
}

type ebml_deserializer = {mutable parent: ebml::doc,
                          mutable pos: uint};

fn mk_ebml_deserializer(d: ebml::doc) -> ebml_deserializer {
    {mutable parent: d, mutable pos: d.start}
}

impl of deserializer for ebml_deserializer {
    fn next_doc(exp_tag: ebml_serializer_tag) -> ebml::doc {
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

    fn next_u64(exp_tag: ebml_serializer_tag) -> u64 {
        let r = ebml::doc_as_vu64(self.next_doc(exp_tag));
        #debug["next_u64 exp_tag=%? result=%?", exp_tag, r];
        ret r;
    }

    fn read_nil() -> () { () }
    fn read_u64() -> u64 { self.next_u64(es_u64) }
    fn read_i64() -> i64 { self.next_u64(es_i64) as i64 }
    fn read_bool() -> bool { self.next_u64(es_bool) as bool }
    fn read_f64() -> f64 { fail "Float"; }
    fn read_str() -> str { ebml::doc_str(self.next_doc(es_str)) }

    // Compound types:
    fn read_enum<T:copy>(_name: str, f: fn() -> T) -> T {
        self.push_doc(self.next_doc(es_enum), f)
    }

    fn read_enum_variant<T:copy>(f: fn(uint) -> T) -> T {
        let idx = self.next_u64(es_enum_vid) as uint;
        self.push_doc(self.next_doc(es_enum_body)) {||
            f(idx)
        }
    }

    fn read_enum_variant_arg<T:copy>(_idx: uint, f: fn() -> T) -> T {
        f()
    }

    fn read_vec<T:copy>(f: fn(uint) -> T) -> T {
        self.push_doc(self.next_doc(es_vec)) {||
            let len = self.next_u64(es_vec_len) as uint;
            f(len)
        }
    }

    fn read_vec_elt<T:copy>(_idx: uint, f: fn() -> T) -> T {
        self.push_doc(self.next_doc(es_vec_elt), f)
    }

    fn read_box<T:copy>(f: fn() -> T) -> T {
        f()
    }

    fn read_uniq<T:copy>(f: fn() -> T) -> T {
        f()
    }

    fn read_rec<T:copy>(f: fn() -> T) -> T {
        f()
    }

    fn read_rec_field<T:copy>(_f_name: str, _f_idx: uint, f: fn() -> T) -> T {
        f()
    }

    fn read_tup<T:copy>(_sz: uint, f: fn() -> T) -> T {
        f()
    }

    fn read_tup_elt<T:copy>(_idx: uint, f: fn() -> T) -> T {
        f()
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
        let ebml_w = ebml::create_writer(io::mem_buffer_writer(mbuf));
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
