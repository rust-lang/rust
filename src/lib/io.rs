import os.libc;

native "rust" mod rustrt {
  fn rust_get_stdin() -> os.libc.FILE;
  fn rust_get_stdout() -> os.libc.FILE;
}

// Reading

// TODO This is all buffered. We might need an unbuffered variant as well

tag seek_style {seek_set; seek_end; seek_cur;}

type reader =
    state obj {
          impure fn read_byte() -> u8;
          impure fn read_bytes(uint len) -> vec[u8];
          impure fn read_char() -> int;
          impure fn unread_char(int i);
          impure fn eof() -> bool;
          impure fn read_line() -> str;
          impure fn read_c_str() -> str;
          impure fn read_le_uint(uint size) -> uint;
          impure fn read_le_int(uint size) -> int;

          impure fn seek(int offset, seek_style whence);
          impure fn tell() -> uint; // TODO: eventually u64
    };

fn convert_whence(seek_style whence) -> int {
    alt (whence) {
        case (seek_set) {ret 0;}
        case (seek_cur) {ret 1;}
        case (seek_end) {ret 2;}
    }
}

state obj FILE_reader(os.libc.FILE f, bool must_close) {
    impure fn read_byte() -> u8 {
        ret os.libc.fgetc(f) as u8;
    }
    impure fn read_bytes(uint len) -> vec[u8] {
        auto buf = _vec.alloc[u8](len);
        auto read = os.libc.fread(_vec.buf[u8](buf), 1u, len, f);
        _vec.len_set[u8](buf, read);
        ret buf;
    }
    impure fn read_char() -> int {
        ret os.libc.fgetc(f);
    }
    impure fn unread_char(int ch) {
        os.libc.ungetc(ch, f);
    }
    impure fn eof() -> bool {
      auto ch = os.libc.fgetc(f);
      if (ch == -1) {ret true;}
      os.libc.ungetc(ch, f);
      ret false;
    }
    impure fn read_line() -> str {
      auto buf = "";
      while (true) {
        auto ch = os.libc.fgetc(f);
        if (ch == -1) { ret buf; }
        if (ch == 10) { ret buf; }
        buf += _str.unsafe_from_bytes(vec(ch as u8));
      }
      ret buf;
    }
    impure fn read_c_str() -> str {
        auto buf = "";
        while (true) {
            auto ch = os.libc.fgetc(f);
            if (ch < 1) { ret buf; }
            buf += _str.unsafe_from_bytes(vec(ch as u8));
        }
        ret buf;
    }
    // TODO deal with eof?
    impure fn read_le_uint(uint size) -> uint {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (os.libc.fgetc(f) as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val;
    }
    impure fn read_le_int(uint size) -> int {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (os.libc.fgetc(f) as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val as int; // TODO does that work?
    }
    impure fn seek(int offset, seek_style whence) {
        check(os.libc.fseek(f, offset, convert_whence(whence)) == 0);
    }
    impure fn tell() -> uint {
        ret os.libc.ftell(f) as uint;
    }
    drop {
        if (must_close) {os.libc.fclose(f);}
    }
}

fn stdin() -> reader {
    ret FILE_reader(rustrt.rust_get_stdin(), false);
}

fn file_reader(str path) -> reader {
    auto f = os.libc.fopen(_str.buf(path), _str.buf("r"));
    if (f as uint == 0u) {
        log "error opening " + path;
        fail;
    }
    ret FILE_reader(f, true);
}

// Writing

// TODO This is all unbuffered. We might need a buffered variant as well

tag fileflag {
    append;
    create;
    truncate;
    none;
}

type buf_writer = state obj {
  fn write(vec[u8] v);

  fn seek(int offset, seek_style whence);
  fn tell() -> uint; // TODO: eventually u64
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
          impure fn write_str(str s);
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

// TODO: fileflags
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

type byte_buf = @rec(mutable vec[mutable u8] buf, mutable uint pos);

state obj byte_buf_writer(byte_buf buf) {
    fn write(vec[u8] v) {
        // TODO: optimize
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
        auto pos = buf.pos as int;
        auto len = _vec.len[mutable u8](buf.buf) as int;
        alt (whence) {
            case (seek_set) { pos = offset;         }
            case (seek_cur) { pos += offset;        }
            case (seek_end) { pos = len + offset;   }
        }

        if (pos < 0) {
            pos = 0;
        } else if (pos > len) {
            pos = len;
        }

        buf.pos = pos as uint;
    }

    fn tell() -> uint { ret buf.pos; }
}

// TODO awkward! it's not possible to implement a writer with an extra method
fn string_writer() -> str_writer {
    // FIXME: yikes, this is bad. Needs fixing of mutable syntax.
    let vec[mutable u8] b = vec(mutable 0u8);
    _vec.pop[mutable u8](b);

    let byte_buf buf = @rec(mutable buf = b, mutable pos = 0u);
    state obj str_writer_wrap(writer wr, byte_buf buf) {
        fn get_writer() -> writer {ret wr;}
        fn get_str() -> str {ret _str.unsafe_from_mutable_bytes(buf.buf);}
    }
    ret str_writer_wrap(new_writer(byte_buf_writer(buf)), buf);
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
