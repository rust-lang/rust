

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html
import option::none;
import option::some;

type ebml_tag = {id: uint, size: uint};

type ebml_state = {ebml_tag: ebml_tag, tag_pos: uint, data_pos: uint};


// TODO: When we have module renaming, make "reader" and "writer" separate
// modules within this file.

// ebml reading
type doc = {data: @[u8], start: uint, end: uint};

fn vint_at(data: [u8], start: uint) -> {val: uint, next: uint} {
    let a = data[start];
    if a & 0x80u8 != 0u8 { ret {val: a & 0x7fu8 as uint, next: start + 1u}; }
    if a & 0x40u8 != 0u8 {
        ret {val: (a & 0x3fu8 as uint) << 8u | (data[start + 1u] as uint),
             next: start + 2u};
    } else if a & 0x20u8 != 0u8 {
        ret {val:
                 (a & 0x1fu8 as uint) << 16u |
                     (data[start + 1u] as uint) << 8u |
                     (data[start + 2u] as uint),
             next: start + 3u};
    } else if a & 0x10u8 != 0u8 {
        ret {val:
                 (a & 0x0fu8 as uint) << 24u |
                     (data[start + 1u] as uint) << 16u |
                     (data[start + 2u] as uint) << 8u |
                     (data[start + 3u] as uint),
             next: start + 4u};
    } else { log_err "vint too big"; fail; }
}

fn new_doc(data: @[u8]) -> doc {
    ret {data: data, start: 0u, end: vec::len::<u8>(*data)};
}

fn doc_at(data: @[u8], start: uint) -> doc {
    let elt_tag = vint_at(*data, start);
    let elt_size = vint_at(*data, elt_tag.next);
    let end = elt_size.next + elt_size.val;
    ret {data: data, start: elt_size.next, end: end};
}

fn maybe_get_doc(d: doc, tg: uint) -> option::t<doc> {
    let pos = d.start;
    while pos < d.end {
        let elt_tag = vint_at(*d.data, pos);
        let elt_size = vint_at(*d.data, elt_tag.next);
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
      none. {
        log_err "failed to find block with tag " + uint::to_str(tg, 10u);
        fail;
      }
    }
}

iter docs(d: doc) -> {tag: uint, doc: doc} {
    let pos = d.start;
    while pos < d.end {
        let elt_tag = vint_at(*d.data, pos);
        let elt_size = vint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        put {tag: elt_tag.val,
             doc: {data: d.data, start: elt_size.next, end: pos}};
    }
}

iter tagged_docs(d: doc, tg: uint) -> doc {
    let pos = d.start;
    while pos < d.end {
        let elt_tag = vint_at(*d.data, pos);
        let elt_size = vint_at(*d.data, elt_tag.next);
        pos = elt_size.next + elt_size.val;
        if elt_tag.val == tg {
            put {data: d.data, start: elt_size.next, end: pos};
        }
    }
}

fn doc_data(d: doc) -> [u8] { ret vec::slice::<u8>(*d.data, d.start, d.end); }

fn be_uint_from_bytes(data: @[u8], start: uint, size: uint) -> uint {
    let sz = size;
    assert (sz <= 4u);
    let val = 0u;
    let pos = start;
    while sz > 0u {
        sz -= 1u;
        val += (data[pos] as uint) << sz * 8u;
        pos += 1u;
    }
    ret val;
}

fn doc_as_uint(d: doc) -> uint {
    ret be_uint_from_bytes(d.data, d.start, d.end - d.start);
}


// ebml writing
type writer = {writer: io::buf_writer, mutable size_positions: [uint]};

fn write_sized_vint(w: io::buf_writer, n: uint, size: uint) {
    let buf: [u8];
    alt size {
      1u { buf = [0x80u8 | (n as u8)]; }
      2u { buf = [0x40u8 | (n >> 8u as u8), n & 0xffu as u8]; }
      3u {
        buf =
            [0x20u8 | (n >> 16u as u8), n >> 8u & 0xffu as u8,
             n & 0xffu as u8];
      }
      4u {
        buf =
            [0x10u8 | (n >> 24u as u8), n >> 16u & 0xffu as u8,
             n >> 8u & 0xffu as u8, n & 0xffu as u8];
      }
      _ { log_err "vint to write too big"; fail; }
    }
    w.write(buf);
}

fn write_vint(w: io::buf_writer, n: uint) {
    if n < 0x7fu { write_sized_vint(w, n, 1u); ret; }
    if n < 0x4000u { write_sized_vint(w, n, 2u); ret; }
    if n < 0x200000u { write_sized_vint(w, n, 3u); ret; }
    if n < 0x10000000u { write_sized_vint(w, n, 4u); ret; }
    log_err "vint to write too big";
    fail;
}

fn create_writer(w: io::buf_writer) -> writer {
    let size_positions: [uint] = [];
    ret {writer: w, mutable size_positions: size_positions};
}


// TODO: Provide a function to write the standard ebml header.
fn start_tag(w: writer, tag_id: uint) {
    // Write the tag ID:

    write_vint(w.writer, tag_id);
    // Write a placeholder four-byte size.

    w.size_positions += [w.writer.tell()];
    let zeroes: [u8] = [0u8, 0u8, 0u8, 0u8];
    w.writer.write(zeroes);
}

fn end_tag(w: writer) {
    let last_size_pos = vec::pop::<uint>(w.size_positions);
    let cur_pos = w.writer.tell();
    w.writer.seek(last_size_pos as int, io::seek_set);
    write_sized_vint(w.writer, cur_pos - last_size_pos - 4u, 4u);
    w.writer.seek(cur_pos as int, io::seek_set);
}
// TODO: optionally perform "relaxations" on end_tag to more efficiently
// encode sizes; this is a fixed point iteration
