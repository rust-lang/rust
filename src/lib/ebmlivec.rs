

// Simple Extensible Binary Markup Language (ebml) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html
import option::none;
import option::some;

type ebml_tag = rec(uint id, uint size);

type ebml_state = rec(ebml_tag ebml_tag, uint tag_pos, uint data_pos);


// TODO: When we have module renaming, make "reader" and "writer" separate
// modules within this file.

// ebml reading
type doc = rec(@u8[] data, uint start, uint end);

fn vint_at(&u8[] data, uint start) -> tup(uint, uint) {
    auto a = data.(start);
    if (a & 0x80u8 != 0u8) { ret tup(a & 0x7fu8 as uint, start + 1u); }
    if (a & 0x40u8 != 0u8) {
        ret tup((a & 0x3fu8 as uint) << 8u | (data.(start + 1u) as uint),
                start + 2u);
    } else if (a & 0x20u8 != 0u8) {
        ret tup((a & 0x1fu8 as uint) << 16u |
                    (data.(start + 1u) as uint) << 8u |
                    (data.(start + 2u) as uint), start + 3u);
    } else if (a & 0x10u8 != 0u8) {
        ret tup((a & 0x0fu8 as uint) << 24u |
                    (data.(start + 1u) as uint) << 16u |
                    (data.(start + 2u) as uint) << 8u |
                    (data.(start + 3u) as uint), start + 4u);
    } else { log_err "vint too big"; fail; }
}

fn new_doc(&@u8[] data) -> doc {
    ret rec(data=data, start=0u, end=ivec::len[u8](*data));
}

fn doc_at(&@u8[] data, uint start) -> doc {
    auto elt_tag = vint_at(*data, start);
    auto elt_size = vint_at(*data, elt_tag._1);
    auto end = elt_size._1 + elt_size._0;
    ret rec(data=data, start=elt_size._1, end=end);
}

fn maybe_get_doc(doc d, uint tg) -> option::t[doc] {
    auto pos = d.start;
    while (pos < d.end) {
        auto elt_tag = vint_at(*d.data, pos);
        auto elt_size = vint_at(*d.data, elt_tag._1);
        pos = elt_size._1 + elt_size._0;
        if (elt_tag._0 == tg) {
            ret some[doc](rec(data=d.data, start=elt_size._1, end=pos));
        }
    }
    ret none[doc];
}

fn get_doc(doc d, uint tg) -> doc {
    alt (maybe_get_doc(d, tg)) {
        case (some(?d)) { ret d; }
        case (none) {
            log_err "failed to find block with tag " + uint::to_str(tg, 10u);
            fail;
        }
    }
}

iter docs(doc d) -> tup(uint, doc) {
    auto pos = d.start;
    while (pos < d.end) {
        auto elt_tag = vint_at(*d.data, pos);
        auto elt_size = vint_at(*d.data, elt_tag._1);
        pos = elt_size._1 + elt_size._0;
        put tup(elt_tag._0, rec(data=d.data, start=elt_size._1, end=pos));
    }
}

iter tagged_docs(doc d, uint tg) -> doc {
    auto pos = d.start;
    while (pos < d.end) {
        auto elt_tag = vint_at(*d.data, pos);
        auto elt_size = vint_at(*d.data, elt_tag._1);
        pos = elt_size._1 + elt_size._0;
        if (elt_tag._0 == tg) {
            put rec(data=d.data, start=elt_size._1, end=pos);
        }
    }
}

fn doc_data(doc d) -> u8[] { ret ivec::slice[u8](*d.data, d.start, d.end); }

fn be_uint_from_bytes(&@u8[] data, uint start, uint size) -> uint {
    auto sz = size;
    assert (sz <= 4u);
    auto val = 0u;
    auto pos = start;
    while (sz > 0u) {
        sz -= 1u;
        val += (data.(pos) as uint) << sz * 8u;
        pos += 1u;
    }
    ret val;
}

fn doc_as_uint(doc d) -> uint {
    ret be_uint_from_bytes(d.data, d.start, d.end - d.start);
}


// ebml writing
type writer = rec(ioivec::buf_writer writer, mutable uint[] size_positions);

fn write_sized_vint(&ioivec::buf_writer w, uint n, uint size) {
    let u8[] buf;
    alt (size) {
        case (1u) { buf = ~[0x80u8 | (n as u8)]; }
        case (2u) { buf = ~[0x40u8 | (n >> 8u as u8), n & 0xffu as u8]; }
        case (3u) {
            buf =
                ~[0x20u8 | (n >> 16u as u8), n >> 8u & 0xffu as u8,
                  n & 0xffu as u8];
        }
        case (4u) {
            buf =
                ~[0x10u8 | (n >> 24u as u8), n >> 16u & 0xffu as u8,
                  n >> 8u & 0xffu as u8, n & 0xffu as u8];
        }
        case (_) { log_err "vint to write too big"; fail; }
    }
    w.write(buf);
}

fn write_vint(&ioivec::buf_writer w, uint n) {
    if (n < 0x7fu) { write_sized_vint(w, n, 1u); ret; }
    if (n < 0x4000u) { write_sized_vint(w, n, 2u); ret; }
    if (n < 0x200000u) { write_sized_vint(w, n, 3u); ret; }
    if (n < 0x10000000u) { write_sized_vint(w, n, 4u); ret; }
    log_err "vint to write too big";
    fail;
}

fn create_writer(&ioivec::buf_writer w) -> writer {
    let uint[] size_positions = ~[];
    ret rec(writer=w, mutable size_positions=size_positions);
}


// TODO: Provide a function to write the standard ebml header.
fn start_tag(&writer w, uint tag_id) {
    // Write the tag ID:

    write_vint(w.writer, tag_id);
    // Write a placeholder four-byte size.

    w.size_positions += ~[w.writer.tell()];
    let u8[] zeroes = ~[0u8, 0u8, 0u8, 0u8];
    w.writer.write(zeroes);
}

fn end_tag(&writer w) {
    auto last_size_pos = ivec::pop[uint](w.size_positions);
    auto cur_pos = w.writer.tell();
    w.writer.seek(last_size_pos as int, ioivec::seek_set);
    write_sized_vint(w.writer, cur_pos - last_size_pos - 4u, 4u);
    w.writer.seek(cur_pos as int, ioivec::seek_set);
}
// TODO: optionally perform "relaxations" on end_tag to more efficiently
// encode sizes; this is a fixed point iteration
