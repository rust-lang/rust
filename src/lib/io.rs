import os.libc;

native "rust" mod rustrt {
  fn rust_get_stdin() -> os.libc.FILE;
  fn rust_get_stdout() -> os.libc.FILE;
}

// Reading

// FIXME This is all buffered. We might need an unbuffered variant as well

tag seek_style {seek_set; seek_end; seek_cur;}

// The raw underlying reader class. All readers must implement this.
type buf_reader =
    state obj {
        impure fn read(uint len) -> vec[u8];
        impure fn unread_byte(int byte);
        impure fn eof() -> bool;

        // FIXME: Seekable really should be orthogonal. We will need
        // inheritance.
        impure fn seek(int offset, seek_style whence);
        impure fn tell() -> uint;
    };

// Convenience methods for reading.
type reader =
    state obj {
          // FIXME: This should inherit from buf_reader.
          impure fn get_buf_reader() -> buf_reader;

          impure fn read_byte() -> int;
          impure fn unread_byte(int byte);
          impure fn read_bytes(uint len) -> vec[u8];
          impure fn read_char() -> char;
          impure fn eof() -> bool;
          impure fn read_line() -> str;
          impure fn read_c_str() -> str;
          impure fn read_le_uint(uint size) -> uint;
          impure fn read_le_int(uint size) -> int;

          impure fn seek(int offset, seek_style whence);
          impure fn tell() -> uint; // FIXME: eventually u64
    };

fn convert_whence(seek_style whence) -> int {
    alt (whence) {
        case (seek_set) {ret 0;}
        case (seek_cur) {ret 1;}
        case (seek_end) {ret 2;}
    }
}

state obj FILE_buf_reader(os.libc.FILE f, bool must_close) {
    impure fn read(uint len) -> vec[u8] {
        auto buf = _vec.alloc[u8](len);
        auto read = os.libc.fread(_vec.buf[u8](buf), 1u, len, f);
        _vec.len_set[u8](buf, read);
        ret buf;
    }
    impure fn unread_byte(int byte) {
        os.libc.ungetc(byte, f);
    }
    impure fn eof() -> bool {
        ret os.libc.feof(f) != 0;
    }
    impure fn seek(int offset, seek_style whence) {
        check (os.libc.fseek(f, offset, convert_whence(whence)) == 0);
    }
    impure fn tell() -> uint {
        ret os.libc.ftell(f) as uint;
    }
    drop {
        if (must_close) { os.libc.fclose(f); }
    }
}

// FIXME: When we have a "self" keyword, move this into read_byte(). This is
// only here so that multiple method implementations below can use it.
//
// FIXME: Return value should be option[u8], not int.
impure fn read_byte_from_buf_reader(buf_reader rdr) -> int {
    auto buf = rdr.read(1u);
    if (_vec.len[u8](buf) == 0u) {
        ret -1;
    }
    ret buf.(0) as int;
}

// FIXME: Convert this into pseudomethods on buf_reader.
state obj new_reader(buf_reader rdr) {
    impure fn get_buf_reader() -> buf_reader {
        ret rdr;
    }
    impure fn read_byte() -> int {
        ret read_byte_from_buf_reader(rdr);
    }
    impure fn unread_byte(int byte) {
        ret rdr.unread_byte(byte);
    }
    impure fn read_bytes(uint len) -> vec[u8] {
        ret rdr.read(len);
    }
    impure fn read_char() -> char {
        auto c0 = read_byte_from_buf_reader(rdr);
        if (c0 == -1) {ret -1 as char;} // FIXME will this stay valid?
        auto b0 = c0 as u8;
        auto w = _str.utf8_char_width(b0);
        check(w > 0u);
        if (w == 1u) {ret b0 as char;}
        auto val = 0u;
        while (w > 1u) {
            w -= 1u;
            auto next = read_byte_from_buf_reader(rdr);
            check(next > -1);
            check(next & 0xc0 == 0x80);
            val <<= 6u;
            val += (next & 0x3f) as uint;
        }
        // See _str.char_at
        val += ((b0 << ((w + 1u) as u8)) as uint) << ((w - 1u) * 6u - w - 1u);
        ret val as char;
    }        
    impure fn eof() -> bool {
        ret rdr.eof();
    }
    impure fn read_line() -> str {
        let vec[u8] buf = vec();
        // No break yet in rustc
        auto go_on = true;
        while (go_on) {
            auto ch = read_byte_from_buf_reader(rdr);
            if (ch == -1 || ch == 10) {go_on = false;}
            else {_vec.push[u8](buf, ch as u8);}
        }
        ret _str.unsafe_from_bytes(buf);
    }
    impure fn read_c_str() -> str {
        let vec[u8] buf = vec();
        auto go_on = true;
        while (go_on) {
            auto ch = read_byte_from_buf_reader(rdr);
            if (ch < 1) {go_on = false;}
            else {_vec.push[u8](buf, ch as u8);}
        }
        ret _str.unsafe_from_bytes(buf);
    }
    // FIXME deal with eof?
    impure fn read_le_uint(uint size) -> uint {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (read_byte_from_buf_reader(rdr) as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val;
    }
    impure fn read_le_int(uint size) -> int {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (read_byte_from_buf_reader(rdr) as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val as int;
    }
    impure fn seek(int offset, seek_style whence) {
        ret rdr.seek(offset, whence);
    }
    impure fn tell() -> uint {
        ret rdr.tell();
    }
}

fn stdin() -> reader {
    ret new_reader(FILE_buf_reader(rustrt.rust_get_stdin(), false));
}

fn file_reader(str path) -> reader {
    auto f = os.libc.fopen(_str.buf(path), _str.buf("r"));
    if (f as uint == 0u) {
        log "error opening " + path;
        fail;
    }
    ret new_reader(FILE_buf_reader(f, true));
}

// FIXME: Remove me once objects are exported.
fn new_reader_(buf_reader bufr) -> reader {
    ret new_reader(bufr);
}


// Byte buffer readers

// TODO: mutable? u8, but this fails with rustboot.
type byte_buf = @rec(vec[u8] buf, mutable uint pos);

state obj byte_buf_reader(byte_buf bbuf) {
    impure fn read(uint len) -> vec[u8] {
        auto rest = _vec.len[u8](bbuf.buf) - bbuf.pos;
        auto to_read = len;
        if (rest < to_read) {
            to_read = rest;
        }
        auto range = _vec.slice[u8](bbuf.buf, bbuf.pos, bbuf.pos + to_read);
        bbuf.pos += to_read;
        ret range;
    }

    impure fn unread_byte(int byte) {
        log "TODO: unread_byte";
        fail;
    }

    impure fn eof() -> bool {
        ret bbuf.pos == _vec.len[u8](bbuf.buf);
    }

    impure fn seek(int offset, seek_style whence) {
        auto pos = bbuf.pos;
        auto len = _vec.len[u8](bbuf.buf);
        bbuf.pos = seek_in_buf(offset, pos, len, whence);
    }

    impure fn tell() -> uint { ret bbuf.pos; }
}

fn new_byte_buf_reader(vec[u8] buf) -> byte_buf_reader {
    ret byte_buf_reader(@rec(buf=buf, mutable pos=0u));
}


// Writing

tag fileflag {
    append;
    create;
    truncate;
    none;
}

type buf_writer = state obj {
  fn write(vec[u8] v);

  // FIXME: Seekable really should be orthogonal. We will need inheritance.
  fn seek(int offset, seek_style whence);
  fn tell() -> uint; // FIXME: eventually u64
};

state obj FILE_writer(os.libc.FILE f, bool must_close) {
    fn write(vec[u8] v) {
        auto len = _vec.len[u8](v);
        auto vbuf = _vec.buf[u8](v);
        auto nout = os.libc.fwrite(vbuf, len, 1u, f);
        if (nout < 1u) {
            log "error dumping buffer";
        }
    }

    fn seek(int offset, seek_style whence) {
        check(os.libc.fseek(f, offset, convert_whence(whence)) == 0);
    }

    fn tell() -> uint {
        ret os.libc.ftell(f) as uint;
    }

    drop {
        if (must_close) {os.libc.fclose(f);}
    }
}

state obj fd_buf_writer(int fd, bool must_close) {
    fn write(vec[u8] v) {
        auto len = _vec.len[u8](v);
        auto count = 0u;
        auto vbuf;
        while (count < len) {
            vbuf = _vec.buf_off[u8](v, count);
            auto nout = os.libc.write(fd, vbuf, len);
            if (nout < 0) {
                log "error dumping buffer";
                log sys.rustrt.last_os_error();
                fail;
            }
            count += nout as uint;
        }
    }

    fn seek(int offset, seek_style whence) {
        log "need 64-bit native calls for seek, sorry";
        fail;
    }

    fn tell() -> uint {
        log "need 64-bit native calls for tell, sorry";
        fail;
    }

    drop {
        if (must_close) {os.libc.close(fd);}
    }
}

fn file_buf_writer(str path, vec[fileflag] flags) -> buf_writer {
    let int fflags =
        os.libc_constants.O_WRONLY() |
        os.libc_constants.O_BINARY();

    for (fileflag f in flags) {
        alt (f) {
            case (append)   { fflags |= os.libc_constants.O_APPEND(); }
            case (create)   { fflags |= os.libc_constants.O_CREAT(); }
            case (truncate) { fflags |= os.libc_constants.O_TRUNC(); }
            case (none) {}
        }
    }

    auto fd = os.libc.open(_str.buf(path),
                           fflags,
                           os.libc_constants.S_IRUSR() |
                           os.libc_constants.S_IWUSR());

    if (fd < 0) {
        log "error opening file for writing";
        log sys.rustrt.last_os_error();
        fail;
    }
    ret fd_buf_writer(fd, true);
}

type writer =
    state obj {
          fn get_buf_writer() -> buf_writer;
          // write_str will continue to do utf-8 output only. an alternative
          // function will be provided for general encoded string output
          impure fn write_str(str s);
          impure fn write_char(char ch);
          impure fn write_int(int n);
          impure fn write_uint(uint n);
          impure fn write_bytes(vec[u8] bytes);
          impure fn write_le_uint(uint n, uint size);
          impure fn write_le_int(int n, uint size);
    };

fn uint_to_le_bytes(uint n, uint size) -> vec[u8] {
    let vec[u8] bytes = vec();
    while (size > 0u) {
        bytes += vec((n & 255u) as u8);
        n >>= 8u;
        size -= 1u;
    }
    ret bytes;
}

state obj new_writer(buf_writer out) {
    fn get_buf_writer() -> buf_writer {
        ret out;
    }
    impure fn write_str(str s) {
        out.write(_str.bytes(s));
    }
    impure fn write_char(char ch) {
        // FIXME needlessly consy
        out.write(_str.bytes(_str.from_char(ch)));
    }
    impure fn write_int(int n) {
        out.write(_str.bytes(_int.to_str(n, 10u)));
    }
    impure fn write_uint(uint n) {
        out.write(_str.bytes(_uint.to_str(n, 10u)));
    }
    impure fn write_bytes(vec[u8] bytes) {
        out.write(bytes);
    }
    impure fn write_le_uint(uint n, uint size) {
        out.write(uint_to_le_bytes(n, size));
    }
    impure fn write_le_int(int n, uint size) {
        out.write(uint_to_le_bytes(n as uint, size));
    }
}

// FIXME: Remove me once objects are exported.
fn new_writer_(buf_writer out) -> writer {
    ret new_writer(out);
}

fn file_writer(str path, vec[fileflag] flags) -> writer {
    ret new_writer(file_buf_writer(path, flags));
}

// FIXME: fileflags
fn buffered_file_buf_writer(str path) -> buf_writer {
    auto f = os.libc.fopen(_str.buf(path), _str.buf("w"));
    if (f as uint == 0u) {
        log "error opening " + path;
        fail;
    }
    ret FILE_writer(f, true);
}

// FIXME it would be great if this could be a const
fn stdout() -> writer {
    ret new_writer(fd_buf_writer(1, false));
}

type str_writer =
    state obj {
          fn get_writer() -> writer;
          fn get_str() -> str;
    };

type mutable_byte_buf = @rec(mutable vec[mutable u8] buf, mutable uint pos);

state obj byte_buf_writer(mutable_byte_buf buf) {
    fn write(vec[u8] v) {
        // FIXME: optimize
        auto vlen = _vec.len[u8](v);
        auto vpos = 0u;
        while (vpos < vlen) {
            auto b = v.(vpos);
            if (buf.pos == _vec.len[mutable u8](buf.buf)) {
                buf.buf += vec(mutable b);
            } else {
                buf.buf.(buf.pos) = b;
            }
            buf.pos += 1u;
            vpos += 1u;
        }
    }

    fn seek(int offset, seek_style whence) {
        auto pos = buf.pos;
        auto len = _vec.len[mutable u8](buf.buf);
        buf.pos = seek_in_buf(offset, pos, len, whence);
    }

    fn tell() -> uint { ret buf.pos; }
}

fn string_writer() -> str_writer {
    // FIXME: yikes, this is bad. Needs fixing of mutable syntax.
    let vec[mutable u8] b = vec(mutable 0u8);
    _vec.pop[mutable u8](b);

    let mutable_byte_buf buf = @rec(mutable buf = b, mutable pos = 0u);
    state obj str_writer_wrap(writer wr, mutable_byte_buf buf) {
        fn get_writer() -> writer {ret wr;}
        fn get_str() -> str {ret _str.unsafe_from_mutable_bytes(buf.buf);}
    }
    ret str_writer_wrap(new_writer(byte_buf_writer(buf)), buf);
}


// Utility functions

fn seek_in_buf(int offset, uint pos, uint len, seek_style whence) -> uint {
    auto bpos = pos as int;
    auto blen = len as int;
    alt (whence) {
        case (seek_set) { bpos = offset;        }
        case (seek_cur) { bpos += offset;       }
        case (seek_end) { bpos = blen + offset; }
    }

    if (bpos < 0) {
        bpos = 0;
    } else if (bpos > blen) {
        bpos = blen;
    }

    ret bpos as uint;
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
