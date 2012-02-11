/*
Module: serialization

Support code for serialization.
*/

import list::list;

iface serializer {
    // Primitive types:
    fn emit_nil();
    fn emit_u64(v: u64);
    fn emit_i64(v: u64);
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

/*
type ppserializer = {
    writer: io::writer
};

impl serializer for ppserializer {
    fn emit_nil() { self.writer.write_str("()") }

    fn emit_u64(v: u64) { self.writer.write_str(#fmt["%lu", v]); }
    fn emit_i64(v: u64) { ebml::write_vint(self, v as uint) }
    fn emit_bool(v: bool) { ebml::write_vint(self, v as uint) }
    fn emit_f64(v: f64) { fail "float serialization not impl"; }
    fn emit_str(v: str) {
        self.wr_tag(es_str as uint) {|| self.wr_str(v) }
    }

    fn emit_enum(name: str, f: fn()) {
        self.wr_tag(es_enum as uint) {|| f() }
    }
    fn emit_enum_variant(v_name: str, v_id: uint, f: fn()) {
        self.wr_tag(es_enum_vid as uint) {|| self.write_vint(v_id) }
        self.wr_tag(es_enum_body as uint) {|| f() }
    }

    fn emit_vec(len: uint, f: fn()) {
        self.wr_tag(es_vec as uint) {||
            self.wr_tag(es_vec_len as uint) {|| self.write_vint(len) }
            f()
        }
    }

    fn emit_vec_elt(idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_vec_elt(idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_box(f: fn()) { f() }
    fn emit_uniq(f: fn()) { f() }
    fn emit_rec_field(f_name: str, f_idx: uint, f: fn()) { f() }
    fn emit_tup(sz: uint, f: fn()) { f() }
    fn emit_tup_elt(idx: uint, f: fn()) { f() }
}
*/

enum ebml_serializer_tags {
    es_str,
    es_enum, es_enum_vid, es_enum_body,
    es_vec, es_vec_len, es_vec_elt
}

impl of serializer for ebml::writer {
    fn emit_nil() {}

    fn emit_u64(v: u64) { ebml::write_vint(self, v) }
    fn emit_i64(v: u64) { ebml::write_vint(self, v as uint) }
    fn emit_bool(v: bool) { ebml::write_vint(self, v as uint) }
    fn emit_f64(v: f64) { fail "float serialization not impl"; }
    fn emit_str(v: str) {
        self.wr_tag(es_str as uint) {|| self.wr_str(v) }
    }

    fn emit_enum(name: str, f: fn()) {
        self.wr_tag(es_enum as uint) {|| f() }
    }
    fn emit_enum_variant(v_name: str, v_id: uint, f: fn()) {
        self.wr_tag(es_enum_vid as uint) {|| self.write_vint(v_id) }
        self.wr_tag(es_enum_body as uint) {|| f() }
    }
    fn emit_enum_variant_arg(idx: uint, f: fn()) { f() }

    fn emit_vec(len: uint, f: fn()) {
        self.wr_tag(es_vec as uint) {||
            self.wr_tag(es_vec_len as uint) {|| self.write_vint(len) }
            f()
        }
    }

    fn emit_vec_elt(idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_vec_elt(idx: uint, f: fn()) {
        self.wr_tag(es_vec_elt as uint) {|| f() }
    }

    fn emit_box(f: fn()) { f() }
    fn emit_uniq(f: fn()) { f() }
    fn emit_rec(f: fn()) { f() }
    fn emit_rec_field(f_name: str, f_idx: uint, f: fn()) { f() }
    fn emit_tup(sz: uint, f: fn()) { f() }
    fn emit_tup_elt(idx: uint, f: fn()) { f() }
}

type ebml_deserializer = {mutable parent: ebml::doc,
                          mutable pos: uint};

fn mk_ebml_deserializer(d: ebml::doc) -> ebml_deserializer {
    {mutable parent: d, mutable pos: 0u}
}

impl of deserializer for ebml_deserializer {
    fn next_doc(exp_tag: uint) -> ebml::doc {
        if self.pos >= self.parent.end {
            fail "no more documents in current node!";
        }
        let (r_tag, r_doc) = ebml::doc_at(self.parent.data, self.pos);
        if r_tag != exp_tag {
            fail #fmt["expected EMBL doc with tag %u but found tag %u",
                      exp_tag, r_tag];
        }
        if r_doc.end >= self.parent.end {
            fail #fmt["invalid EBML, child extends to 0x%x, parent to 0x%x",
                      r_doc.end, self.parent.end];
        }
        self.pos = result.end;
        ret result;
    }

    fn push_doc<T: copy>(d: ebml::doc, f: fn() -> T) -> T{
        let old_parent = self.parent;
        let old_pos = self.pos;
        self.parent = d;
        self.pos = 0u;
        let r = f();
        self.parent = old_parent;
        self.pos = old_pos;
        ret r;
    }

    fn next_u64(exp_tag: uint) {
        ebml::doc_as_uint(self.next_doc(exp_tag))
    }

    fn read_nil() -> () { () }
    fn read_u64() -> u64 { next_u64(es_u64) }
    fn read_i64() -> i64 { next_u64(es_u64) as i64 }
    fn read_bool() -> bool { next_u64(es_u64) as bool }
    fn read_f64() -> f64 { fail "Float"; }
    fn read_str() -> str { ebml::doc_str(self.next_doc(es_str)) }

    // Compound types:
    fn read_enum<T:copy>(name: str, f: fn() -> T) -> T {
        self.push_doc(self.next_doc(es_enum), f)
    }

    fn read_enum_variant<T:copy>(f: fn(uint) -> T) -> T {
        let idx = self.next_u64(es_enum_vid);
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

    fn read_vec_elt<T:copy>(idx: uint, f: fn() -> T) -> T {
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

    fn read_rec_field<T:copy>(f_name: str, f_idx: uint, f: fn() -> T) -> T {
        f()
    }

    fn read_tup<T:copy>(sz: uint, f: fn() -> T) -> T {
        f()
    }

    fn read_tup_elt<T:copy>(idx: uint, f: fn() -> T) -> T {
        f()
    }
}

// ___________________________________________________________________________
// Testing

