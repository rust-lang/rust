

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html
import core::option;
import option::{some, none};

export doc;

export new_doc;
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

type ebml_tag = {id: uint, size: uint};

type ebml_state = {ebml_tag: ebml_tag, tag_pos: uint, data_pos: uint};

// TODO: When we have module renaming, make "reader" and "writer" separate
// modules within this file.

// ebml reading
type doc = {data: @[u8], start: uint, end: uint};

type tagged_doc = {tag: uint, doc: doc};

fn vuint_at(data: [u8], start: uint) -> {val: uint, next: uint} {
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

fn new_doc(data: @[u8]) -> doc {
    ret {data: data, start: 0u, end: vec::len::<u8>(*data)};
}

fn doc_at(data: @[u8], start: uint) -> tagged_doc {
    let elt_tag = vuint_at(*data, start);
    let elt_size = vuint_at(*data, elt_tag.next);
    let end = elt_size.next + elt_size.val;
    ret {tag: elt_tag.val,
         doc: {data: data, start: elt_size.next, end: end}};
}

fn maybe_get_doc(d: doc, tg: uint) -> option<doc> {
    let pos = d.start;
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
    let pos = d.start;
    while pos < d.end {
        let elt_tag = vuint_at(*d.data, pos);
        let elt_size = vuint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        it(elt_tag.val, {data: d.data, start: elt_size.next, end: pos});
    }
}

fn tagged_docs(d: doc, tg: uint, it: fn(doc)) {
    let pos = d.start;
    while pos < d.end {
        let elt_tag = vuint_at(*d.data, pos);
        let elt_size = vuint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        if elt_tag.val == tg {
            it({data: d.data, start: elt_size.next, end: pos});
        }
    }
}

fn doc_data(d: doc) -> [u8] { ret vec::slice::<u8>(*d.data, d.start, d.end); }

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
type writer = {writer: io::writer, mutable size_positions: [uint]};

fn write_sized_vuint(w: io::writer, n: uint, size: uint) {
    let buf: [u8];
    alt size {
      1u { buf = [0x80u8 | (n as u8)]; }
      2u { buf = [0x40u8 | ((n >> 8_u) as u8), n as u8]; }
      3u {
        buf = [0x20u8 | ((n >> 16_u) as u8), (n >> 8_u) as u8,
               n as u8];
      }
      4u {
        buf = [0x10u8 | ((n >> 24_u) as u8), (n >> 16_u) as u8,
               (n >> 8_u) as u8, n as u8];
      }
      _ { fail #fmt("vint to write too big: %?", n); }
    }
    w.write(buf);
}

fn write_vuint(w: io::writer, n: uint) {
    if n < 0x7f_u { write_sized_vuint(w, n, 1u); ret; }
    if n < 0x4000_u { write_sized_vuint(w, n, 2u); ret; }
    if n < 0x200000_u { write_sized_vuint(w, n, 3u); ret; }
    if n < 0x10000000_u { write_sized_vuint(w, n, 4u); ret; }
    fail #fmt("vint to write too big: %?", n);
}

fn writer(w: io::writer) -> writer {
    let size_positions: [uint] = [];
    ret {writer: w, mutable size_positions: size_positions};
}

// TODO: Provide a function to write the standard ebml header.
impl writer for writer {
    fn start_tag(tag_id: uint) {
        #debug["Start tag %u", tag_id];

        // Write the enum ID:
        write_vuint(self.writer, tag_id);

        // Write a placeholder four-byte size.
        self.size_positions += [self.writer.tell()];
        let zeroes: [u8] = [0u8, 0u8, 0u8, 0u8];
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

    fn wr_tagged_bytes(tag_id: uint, b: [u8]) {
        write_vuint(self.writer, tag_id);
        write_vuint(self.writer, vec::len(b));
        self.writer.write(b);
    }

    fn wr_tagged_u64(tag_id: uint, v: u64) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v, 8u));
    }

    fn wr_tagged_u32(tag_id: uint, v: u32) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v as u64, 4u));
    }

    fn wr_tagged_u16(tag_id: uint, v: u16) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v as u64, 2u));
    }

    fn wr_tagged_u8(tag_id: uint, v: u8) {
        self.wr_tagged_bytes(tag_id, [v]);
    }

    fn wr_tagged_i64(tag_id: uint, v: i64) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v as u64, 8u));
    }

    fn wr_tagged_i32(tag_id: uint, v: i32) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v as u64, 4u));
    }

    fn wr_tagged_i16(tag_id: uint, v: i16) {
        self.wr_tagged_bytes(tag_id, io::u64_to_be_bytes(v as u64, 2u));
    }

    fn wr_tagged_i8(tag_id: uint, v: i8) {
        self.wr_tagged_bytes(tag_id, [v as u8]);
    }

    fn wr_tagged_str(tag_id: uint, v: str) {
        // Lame: can't use str::as_bytes() here because the resulting
        // vector is NULL-terminated.  Annoyingly, the underlying
        // writer interface doesn't permit us to write a slice of a
        // vector.  We need first-class slices, I think.

        // str::as_bytes(v) {|b| self.wr_tagged_bytes(tag_id, b); }
        self.wr_tagged_bytes(tag_id, str::bytes(v));
    }

    fn wr_bytes(b: [u8]) {
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
