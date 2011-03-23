// Simple Extensible Binary Markup Language (EBML) reader and writer on a
// cursor model. See the specification here:
//     http://www.matroska.org/technical/specs/rfc/index.html

import option.none;
import option.some;

type ebml_tag = rec(uint id, uint size);
type ebml_state = rec(ebml_tag ebml_tag, uint pos);

// TODO: When we have module renaming, make "reader" and "writer" separate
// modules within this file.

// EBML reading

type reader = rec(
    io.reader reader,
    mutable vec[ebml_state] states,
    uint size
);

// TODO: eventually use u64 or big here
impure fn read_vint(&io.reader reader) -> uint {
    auto a = reader.read_byte();
    if (a & 0x80u8 != 0u8) { ret (a & 0x7fu8) as uint; }
    auto b = reader.read_byte();
    if (a & 0x40u8 != 0u8) {
        ret (((a & 0x3fu8) as uint) << 8u) | (b as uint);
    }
    auto c = reader.read_byte();
    if (a & 0x20u8 != 0u8) {
        ret (((a & 0x1fu8) as uint) << 16u) | ((b as uint) << 8u) |
            (c as uint);
    }
    auto d = reader.read_byte();
    if (a & 0x10u8 != 0u8) {
        ret (((a & 0x0fu8) as uint) << 24u) | ((b as uint) << 16u) |
            ((c as uint) << 8u) | (d as uint);
    }

    log "vint too big"; fail;
}

impure fn create_reader(&io.reader r) -> reader {
    let vec[ebml_state] states = vec();

    // Calculate the size of the stream.
    auto pos = r.tell();
    r.seek(0, io.seek_end);
    auto size = r.tell() - pos;
    r.seek(pos as int, io.seek_set);

    ret rec(reader=r, mutable states=states, size=size);
}

impure fn bytes_left(&reader r) -> uint {
    auto pos = r.reader.tell();
    alt (_vec.last[ebml_state](r.states)) {
        case (none[ebml_state])      { ret r.size - pos; }
        case (some[ebml_state](?st)) { ret st.pos + st.ebml_tag.size - pos; }
    }
}

impure fn read_tag(&reader r) -> ebml_tag {
    auto id = read_vint(r.reader);
    auto size = read_vint(r.reader);
    ret rec(id=id, size=size);
}

// Reads a tag and moves the cursor to its first child or data segment.
impure fn move_to_first_child(&reader r) {
    auto pos = r.reader.tell();
    auto t = read_tag(r);
    r.states += vec(rec(ebml_tag=t, pos=pos));
}

// Reads a tag and skips over its contents, moving to its next sibling.
impure fn move_to_next_sibling(&reader r) {
    auto t = read_tag(r);
    r.reader.seek(t.size as int, io.seek_cur);
}

// Moves to the parent of this tag.
impure fn move_to_parent(&reader r) {
    check (_vec.len[ebml_state](r.states) > 0u);
    auto st = _vec.pop[ebml_state](r.states);
    r.reader.seek(st.pos as int, io.seek_set);
}

// Reads the data segment of a tag.
impure fn read_data(&reader r) -> vec[u8] {
    ret r.reader.read_bytes(bytes_left(r));
}

impure fn peek(&reader r) -> ebml_tag {
    check (bytes_left(r) > 0u);
    auto pos = r.reader.tell();
    auto t = read_tag(r);
    r.reader.seek(pos as int, io.seek_set);
    ret t;
}


// EBML writing

type writer = rec(io.buf_writer writer, mutable vec[uint] size_positions);

fn write_sized_vint(&io.buf_writer w, uint n, uint size) {
    let vec[u8] buf;
    alt (size) {
        case (1u) {
            buf = vec(0x80u8 | (n as u8));
        }
        case (2u) {
            buf = vec(0x40u8 | ((n >> 8u) as u8),
                      (n & 0xffu) as u8);
        }
        case (3u) {
            buf = vec(0x20u8 | ((n >> 16u) as u8),
                      ((n >> 8u) & 0xffu) as u8,
                      (n & 0xffu) as u8);
        }
        case (4u) {
            buf = vec(0x10u8 | ((n >> 24u) as u8),
                      ((n >> 16u) & 0xffu) as u8,
                      ((n >> 8u) & 0xffu) as u8,
                      (n & 0xffu) as u8);
        }
        case (_) {
            log "vint to write too big";
            fail;
        }
    }

    w.write(buf);
}

fn write_vint(&io.buf_writer w, uint n) {
    if (n < 0x7fu)          { write_sized_vint(w, n, 1u); ret; }
    if (n < 0x4000u)        { write_sized_vint(w, n, 2u); ret; }
    if (n < 0x200000u)      { write_sized_vint(w, n, 3u); ret; }
    if (n < 0x10000000u)    { write_sized_vint(w, n, 4u); ret; }
    log "vint to write too big";
    fail;
}

fn create_writer(&io.buf_writer w) -> writer {
    let vec[uint] size_positions = vec();
    ret rec(writer=w, mutable size_positions=size_positions);
}

// TODO: Provide a function to write the standard EBML header.

fn start_tag(&writer w, uint tag_id) {
    // Write the tag ID.
    write_vint(w.writer, tag_id);

    // Write a placeholder four-byte size.
    w.size_positions += vec(w.writer.tell());
    let vec[u8] zeroes = vec(0u8, 0u8, 0u8, 0u8);
    w.writer.write(zeroes);
}

fn end_tag(&writer w) {
    auto last_size_pos = _vec.pop[uint](w.size_positions);
    auto cur_pos = w.writer.tell();
    w.writer.seek(last_size_pos as int, io.seek_set);
    write_sized_vint(w.writer, cur_pos - last_size_pos - 4u, 4u);
    w.writer.seek(cur_pos as int, io.seek_set);
}

// TODO: optionally perform "relaxations" on end_tag to more efficiently
// encode sizes; this is a fixed point iteration

