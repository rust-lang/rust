

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html
import core::option;
import option::{some, none};

export doc;
export doc_at;
export maybe_get_doc;
export get_doc;
export docs;
export tagged_docs;
export doc_data;
export doc_as_str;
export doc_as_u8;
export doc_as_u16;
export doc_as_u32;
export doc_as_u64;
export doc_as_i8;
export doc_as_i16;
export doc_as_i32;
export doc_as_i64;
export writer;
export serializer;
export ebml_deserializer;
export deserializer;
export with_doc_data;

type ebml_tag = {id: uint, size: uint};

type ebml_state = {ebml_tag: ebml_tag, tag_pos: uint, data_pos: uint};

// TODO: When we have module renaming, make "reader" and "writer" separate
// modules within this file.

// ebml reading
type doc = {data: @~[u8], start: uint, end: uint};

type tagged_doc = {tag: uint, doc: doc};

fn vuint_at(data: &[u8], start: uint) -> {val: uint, next: uint} {
    let a = data[start];
    if a & 0x80u8 != 0u8 {
        ret {val: (a & 0x7fu8) as uint, next: start + 1u};
    }
    if a & 0x40u8 != 0u8 {
        ret {val: ((a & 0x3fu8) as uint) << 8u |
                 (data[start + 1u] as uint),
             next: start + 2u};
    } else if a & 0x20u8 != 0u8 {
        ret {val: ((a & 0x1fu8) as uint) << 16u |
                 (data[start + 1u] as uint) << 8u |
                 (data[start + 2u] as uint),
             next: start + 3u};
    } else if a & 0x10u8 != 0u8 {
        ret {val: ((a & 0x0fu8) as uint) << 24u |
                 (data[start + 1u] as uint) << 16u |
                 (data[start + 2u] as uint) << 8u |
                 (data[start + 3u] as uint),
             next: start + 4u};
    } else { #error("vint too big"); fail; }
}

fn doc(data: @~[u8]) -> doc {
    ret {data: data, start: 0u, end: vec::len::<u8>(*data)};
}

fn doc_at(data: @~[u8], start: uint) -> tagged_doc {
    let elt_tag = vuint_at(*data, start);
    let elt_size = vuint_at(*data, elt_tag.next);
    let end = elt_size.next + elt_size.val;
    ret {tag: elt_tag.val,
         doc: {data: data, start: elt_size.next, end: end}};
}

fn maybe_get_doc(d: doc, tg: uint) -> option<doc> {
    let mut pos = d.start;
    while pos < d.end {
        let elt_tag = vuint_at(*d.data, pos);
        let elt_size = vuint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        if elt_tag.val == tg {
            ret some::<doc>({data: d.data, start: elt_size.next, end: pos});
        }
    }
    ret none::<doc>;
}

fn get_doc(d: doc, tg: uint) -> doc {
    alt maybe_get_doc(d, tg) {
      some(d) { ret d; }
      none {
        #error("failed to find block with tag %u", tg);
        fail;
      }
    }
}

fn docs(d: doc, it: fn(uint, doc)) {
    let mut pos = d.start;
    while pos < d.end {
        let elt_tag = vuint_at(*d.data, pos);
        let elt_size = vuint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        it(elt_tag.val, {data: d.data, start: elt_size.next, end: pos});
    }
}

fn tagged_docs(d: doc, tg: uint, it: fn(doc)) {
    let mut pos = d.start;
    while pos < d.end {
        let elt_tag = vuint_at(*d.data, pos);
        let elt_size = vuint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        if elt_tag.val == tg {
            it({data: d.data, start: elt_size.next, end: pos});
        }
    }
}

fn doc_data(d: doc) -> ~[u8] { vec::slice::<u8>(*d.data, d.start, d.end) }

fn with_doc_data<T>(d: doc, f: fn(x:&[u8]) -> T) -> T {
    ret f(vec::view::<u8>(*d.data, d.start, d.end));
}

fn doc_as_str(d: doc) -> str { ret str::from_bytes(doc_data(d)); }

fn doc_as_u8(d: doc) -> u8 {
    assert d.end == d.start + 1u;
    ret (*d.data)[d.start];
}

fn doc_as_u16(d: doc) -> u16 {
    assert d.end == d.start + 2u;
    ret io::u64_from_be_bytes(*d.data, d.start, 2u) as u16;
}

fn doc_as_u32(d: doc) -> u32 {
    assert d.end == d.start + 4u;
    ret io::u64_from_be_bytes(*d.data, d.start, 4u) as u32;
}

fn doc_as_u64(d: doc) -> u64 {
    assert d.end == d.start + 8u;
    ret io::u64_from_be_bytes(*d.data, d.start, 8u);
}

fn doc_as_i8(d: doc) -> i8 { doc_as_u8(d) as i8 }
fn doc_as_i16(d: doc) -> i16 { doc_as_u16(d) as i16 }
fn doc_as_i32(d: doc) -> i32 { doc_as_u32(d) as i32 }
fn doc_as_i64(d: doc) -> i64 { doc_as_u64(d) as i64 }

// ebml writing
type writer = {writer: io::writer, mut size_positions: ~[uint]};

fn write_sized_vuint(w: io::writer, n: uint, size: uint) {
    alt size {
      1u {
        w.write(&[0x80u8 | (n as u8)]);
      }
      2u {
        w.write(&[0x40u8 | ((n >> 8_u) as u8), n as u8]);
      }
      3u {
        w.write(&[0x20u8 | ((n >> 16_u) as u8), (n >> 8_u) as u8,
                 n as u8]);
      }
      4u {
        w.write(&[0x10u8 | ((n >> 24_u) as u8), (n >> 16_u) as u8,
                 (n >> 8_u) as u8, n as u8]);
      }
      _ { fail #fmt("vint to write too big: %?", n); }
    };
}

fn write_vuint(w: io::writer, n: uint) {
    if n < 0x7f_u { write_sized_vuint(w, n, 1u); ret; }
    if n < 0x4000_u { write_sized_vuint(w, n, 2u); ret; }
    if n < 0x200000_u { write_sized_vuint(w, n, 3u); ret; }
    if n < 0x10000000_u { write_sized_vuint(w, n, 4u); ret; }
    fail #fmt("vint to write too big: %?", n);
}

fn writer(w: io::writer) -> writer {
    let size_positions: ~[uint] = ~[];
    ret {writer: w, mut size_positions: size_positions};
}

// TODO: Provide a function to write the standard ebml header.
impl writer for writer {
    fn start_tag(tag_id: uint) {
        #debug["Start tag %u", tag_id];

        // Write the enum ID:
        write_vuint(self.writer, tag_id);

        // Write a placeholder four-byte size.
        vec::push(self.size_positions, self.writer.tell());
        let zeroes: &[u8] = &[0u8, 0u8, 0u8, 0u8];
        self.writer.write(zeroes);
    }

    fn end_tag() {
        let last_size_pos = vec::pop::<uint>(self.size_positions);
        let cur_pos = self.writer.tell();
        self.writer.seek(last_size_pos as int, io::seek_set);
        let size = (cur_pos - last_size_pos - 4u);
        write_sized_vuint(self.writer, size, 4u);
        self.writer.seek(cur_pos as int, io::seek_set);

        #debug["End tag (size = %u)", size];
    }

    fn wr_tag(tag_id: uint, blk: fn()) {
        self.start_tag(tag_id);
        blk();
        self.end_tag();
    }

    fn wr_tagged_bytes(tag_id: uint, b: &[u8]) {
        write_vuint(self.writer, tag_id);
        write_vuint(self.writer, vec::len(b));
        self.writer.write(b);
    }

    fn wr_tagged_u64(tag_id: uint, v: u64) {
        do io::u64_to_be_bytes(v, 8u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_u32(tag_id: uint, v: u32) {
        do io::u64_to_be_bytes(v as u64, 4u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_u16(tag_id: uint, v: u16) {
        do io::u64_to_be_bytes(v as u64, 2u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_u8(tag_id: uint, v: u8) {
        self.wr_tagged_bytes(tag_id, &[v]);
    }

    fn wr_tagged_i64(tag_id: uint, v: i64) {
        do io::u64_to_be_bytes(v as u64, 8u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_i32(tag_id: uint, v: i32) {
        do io::u64_to_be_bytes(v as u64, 4u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_i16(tag_id: uint, v: i16) {
        do io::u64_to_be_bytes(v as u64, 2u) |v| {
            self.wr_tagged_bytes(tag_id, v);
        }
    }

    fn wr_tagged_i8(tag_id: uint, v: i8) {
        self.wr_tagged_bytes(tag_id, &[v as u8]);
    }

    fn wr_tagged_str(tag_id: uint, v: str) {
        // Lame: can't use str::as_bytes() here because the resulting
        // vector is NULL-terminated.  Annoyingly, the underlying
        // writer interface doesn't permit us to write a slice of a
        // vector.  We need first-class slices, I think.

        // str::as_bytes(v) {|b| self.wr_tagged_bytes(tag_id, b); }
        self.wr_tagged_bytes(tag_id, str::bytes(v));
    }

    fn wr_bytes(b: &[u8]) {
        #debug["Write %u bytes", vec::len(b)];
        self.writer.write(b);
    }

    fn wr_str(s: str) {
        #debug["Write str: %?", s];
        self.writer.write(str::bytes(s));
    }
}

// TODO: optionally perform "relaxations" on end_tag to more efficiently
// encode sizes; this is a fixed point iteration

// Set to true to generate more debugging in EBML serialization.
// Totally lame approach.
const debug: bool = true;

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

impl serializer of serialization::serializer for ebml::writer {
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
        do self.wr_tag(es_vec as uint) {
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

type ebml_deserializer = {mut parent: ebml::doc,
                          mut pos: uint};

fn ebml_deserializer(d: ebml::doc) -> ebml_deserializer {
    {mut parent: d, mut pos: d.start}
}

impl deserializer of serialization::deserializer for ebml_deserializer {
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
               copy self.parent.start, copy self.parent.end,
               copy self.pos, r_tag, r_doc.start, r_doc.end];
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
        if v > (core::uint::max_value as u64) {
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
        do self.push_doc(self.next_doc(es_enum_body)) {
            f(idx)
        }
    }

    fn read_enum_variant_arg<T:copy>(idx: uint, f: fn() -> T) -> T {
        #debug["read_enum_variant_arg(idx=%u)", idx];
        f()
    }

    fn read_vec<T:copy>(f: fn(uint) -> T) -> T {
        #debug["read_vec()"];
        do self.push_doc(self.next_doc(es_vec)) {
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
// Testing

#[test]
fn test_option_int() {
    fn serialize_1<S: serialization::serializer>(s: S, v: int) {
        s.emit_i64(v as i64);
    }

    fn serialize_0<S: serialization::serializer>(s: S, v: option<int>) {
        do s.emit_enum("core::option::t") {
            alt v {
              none {
                s.emit_enum_variant("core::option::none", 0u, 0u, || { } );
              }
              some(v0) {
                do s.emit_enum_variant("core::option::some", 1u, 1u) {
                    s.emit_enum_variant_arg(0u, || serialize_1(s, v0));
                }
              }
            }
        }
    }

    fn deserialize_1<S: serialization::deserializer>(s: S) -> int {
        s.read_i64() as int
    }

    fn deserialize_0<S: serialization::deserializer>(s: S) -> option<int> {
        do s.read_enum("core::option::t") {
            do s.read_enum_variant |i| {
                alt check i {
                  0u { none }
                  1u {
                    let v0 = do s.read_enum_variant_arg(0u) {
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
        let mbuf = io::mem_buffer();
        let ebml_w = ebml::writer(io::mem_buffer_writer(mbuf));
        serialize_0(ebml_w, v);
        let ebml_doc = ebml::doc(@io::mem_buffer_buf(mbuf));
        let deser = ebml_deserializer(ebml_doc);
        let v1 = deserialize_0(deser);
        #debug["v1 == %?", v1];
        assert v == v1;
    }

    test_v(some(22));
    test_v(none);
    test_v(some(3));
}
