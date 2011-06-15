
import os::libc;

native "rust" mod rustrt {
    fn rust_get_stdin() -> os::libc::FILE;
    fn rust_get_stdout() -> os::libc::FILE;
}


// Reading

// FIXME This is all buffered. We might need an unbuffered variant as well
tag seek_style { seek_set; seek_end; seek_cur; }


// The raw underlying reader class. All readers must implement this.
type buf_reader =
    obj {
        fn read(uint) -> vec[u8] ;
        fn read_byte() -> int ;
        fn unread_byte(int) ;
        fn eof() -> bool ;

            // FIXME: Seekable really should be orthogonal. We will need
            // inheritance.
            fn seek(int, seek_style) ;
        fn tell() -> uint ;
    };


// Convenience methods for reading.
type reader =
    obj {

            // FIXME: This should inherit from buf_reader.
            fn get_buf_reader() -> buf_reader ;
        fn read_byte() -> int ;
        fn unread_byte(int) ;
        fn read_bytes(uint) -> vec[u8] ;
        fn read_char() -> char ;
        fn eof() -> bool ;
        fn read_line() -> str ;
        fn read_c_str() -> str ;
        fn read_le_uint(uint) -> uint ;
        fn read_le_int(uint) -> int ;
        fn read_be_uint(uint) -> uint ;
        fn read_whole_stream() -> vec[u8] ;
        fn seek(int, seek_style) ;
        fn tell() -> uint ; // FIXME: eventually u64

    };

fn convert_whence(seek_style whence) -> int {
    ret alt (whence) {
            case (seek_set) { 0 }
            case (seek_cur) { 1 }
            case (seek_end) { 2 }
        };
}

obj FILE_buf_reader(os::libc::FILE f, bool must_close) {
    fn read(uint len) -> vec[u8] {
        auto buf = vec::alloc[u8](len);
        auto read = os::libc::fread(vec::buf[u8](buf), 1u, len, f);
        vec::len_set[u8](buf, read);
        ret buf;
    }
    fn read_byte() -> int { ret os::libc::fgetc(f); }
    fn unread_byte(int byte) { os::libc::ungetc(byte, f); }
    fn eof() -> bool { ret os::libc::feof(f) != 0; }
    fn seek(int offset, seek_style whence) {
        assert (os::libc::fseek(f, offset, convert_whence(whence)) == 0);
    }
    fn tell() -> uint {
        ret os::libc::ftell(f) as uint;
    }drop { if (must_close) { os::libc::fclose(f); } }
}


// FIXME: Convert this into pseudomethods on buf_reader.
obj new_reader(buf_reader rdr) {
    fn get_buf_reader() -> buf_reader { ret rdr; }
    fn read_byte() -> int { ret rdr.read_byte(); }
    fn unread_byte(int byte) { ret rdr.unread_byte(byte); }
    fn read_bytes(uint len) -> vec[u8] { ret rdr.read(len); }
    fn read_char() -> char {
        auto c0 = rdr.read_byte();
        if (c0 == -1) {
            ret -1 as char; // FIXME will this stay valid?

        }
        auto b0 = c0 as u8;
        auto w = str::utf8_char_width(b0);
        assert (w > 0u);
        if (w == 1u) { ret b0 as char; }
        auto val = 0u;
        while (w > 1u) {
            w -= 1u;
            auto next = rdr.read_byte();
            assert (next > -1);
            assert (next & 192 == 128);
            val <<= 6u;
            val += next & 63 as uint;
        }
        // See str::char_at

        val += (b0 << (w + 1u as u8) as uint) << (w - 1u) * 6u - w - 1u;
        ret val as char;
    }
    fn eof() -> bool { ret rdr.eof(); }
    fn read_line() -> str {
        let vec[u8] buf = [];
        // No break yet in rustc

        auto go_on = true;
        while (go_on) {
            auto ch = rdr.read_byte();
            if (ch == -1 || ch == 10) {
                go_on = false;
            } else { vec::push[u8](buf, ch as u8); }
        }
        ret str::unsafe_from_bytes(buf);
    }
    fn read_c_str() -> str {
        let vec[u8] buf = [];
        auto go_on = true;
        while (go_on) {
            auto ch = rdr.read_byte();
            if (ch < 1) {
                go_on = false;
            } else { vec::push[u8](buf, ch as u8); }
        }
        ret str::unsafe_from_bytes(buf);
    }

    // FIXME deal with eof?
    fn read_le_uint(uint size) -> uint {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (rdr.read_byte() as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val;
    }
    fn read_le_int(uint size) -> int {
        auto val = 0u;
        auto pos = 0u;
        while (size > 0u) {
            val += (rdr.read_byte() as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val as int;
    }

    // FIXME deal with eof?
    fn read_be_uint(uint size) -> uint {
        auto val = 0u;
        auto sz = size; // FIXME: trans::ml bug workaround

        while (sz > 0u) {
            sz -= 1u;
            val += (rdr.read_byte() as uint) << sz * 8u;
        }
        ret val;
    }
    fn read_whole_stream() -> vec[u8] {
        let vec[u8] buf = [];
        while (!rdr.eof()) { buf += rdr.read(2048u); }
        ret buf;
    }
    fn seek(int offset, seek_style whence) { ret rdr.seek(offset, whence); }
    fn tell() -> uint { ret rdr.tell(); }
}

fn stdin() -> reader {
    ret new_reader(FILE_buf_reader(rustrt::rust_get_stdin(), false));
}

fn file_reader(str path) -> reader {
    auto f = os::libc::fopen(str::buf(path), str::buf("r"));
    if (f as uint == 0u) { log_err "error opening " + path; fail; }
    ret new_reader(FILE_buf_reader(f, true));
}


// FIXME: Remove me once objects are exported.
fn new_reader_(buf_reader bufr) -> reader { ret new_reader(bufr); }


// Byte buffer readers

// TODO: mutable? u8, but this fails with rustboot.
type byte_buf = @rec(vec[u8] buf, mutable uint pos);

obj byte_buf_reader(byte_buf bbuf) {
    fn read(uint len) -> vec[u8] {
        auto rest = vec::len[u8](bbuf.buf) - bbuf.pos;
        auto to_read = len;
        if (rest < to_read) { to_read = rest; }
        auto range = vec::slice[u8](bbuf.buf, bbuf.pos, bbuf.pos + to_read);
        bbuf.pos += to_read;
        ret range;
    }
    fn read_byte() -> int {
        if (bbuf.pos == vec::len[u8](bbuf.buf)) { ret -1; }
        auto b = bbuf.buf.(bbuf.pos);
        bbuf.pos += 1u;
        ret b as int;
    }
    fn unread_byte(int byte) { log_err "TODO: unread_byte"; fail; }
    fn eof() -> bool { ret bbuf.pos == vec::len[u8](bbuf.buf); }
    fn seek(int offset, seek_style whence) {
        auto pos = bbuf.pos;
        auto len = vec::len[u8](bbuf.buf);
        bbuf.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { ret bbuf.pos; }
}

fn new_byte_buf_reader(vec[u8] buf) -> byte_buf_reader {
    ret byte_buf_reader(@rec(buf=buf, mutable pos=0u));
}


// Writing
tag fileflag { append; create; truncate; none; }

type buf_writer =
    obj {
        fn write(vec[u8]) ;

            // FIXME: Seekable really should be orthogonal. We will need
            // inheritance.
            fn seek(int, seek_style) ;
        fn tell() -> uint ; // FIXME: eventually u64

    };

obj FILE_writer(os::libc::FILE f, bool must_close) {
    fn write(vec[u8] v) {
        auto len = vec::len[u8](v);
        auto vbuf = vec::buf[u8](v);
        auto nout = os::libc::fwrite(vbuf, len, 1u, f);
        if (nout < 1u) { log_err "error dumping buffer"; }
    }
    fn seek(int offset, seek_style whence) {
        assert (os::libc::fseek(f, offset, convert_whence(whence)) == 0);
    }
    fn tell() -> uint {
        ret os::libc::ftell(f) as uint;
    }drop { if (must_close) { os::libc::fclose(f); } }
}

obj fd_buf_writer(int fd, bool must_close) {
    fn write(vec[u8] v) {
        auto len = vec::len[u8](v);
        auto count = 0u;
        auto vbuf;
        while (count < len) {
            vbuf = vec::buf_off[u8](v, count);
            auto nout = os::libc::write(fd, vbuf, len);
            if (nout < 0) {
                log_err "error dumping buffer";
                log_err sys::rustrt::last_os_error();
                fail;
            }
            count += nout as uint;
        }
    }
    fn seek(int offset, seek_style whence) {
        log_err "need 64-bit native calls for seek, sorry";
        fail;
    }
    fn tell() -> uint {
        log_err "need 64-bit native calls for tell, sorry";
        fail;
    }drop { if (must_close) { os::libc::close(fd); } }
}

fn file_buf_writer(str path, vec[fileflag] flags) -> buf_writer {
    let int fflags =
        os::libc_constants::O_WRONLY() | os::libc_constants::O_BINARY();
    for (fileflag f in flags) {
        alt (f) {
            case (append) { fflags |= os::libc_constants::O_APPEND(); }
            case (create) { fflags |= os::libc_constants::O_CREAT(); }
            case (truncate) { fflags |= os::libc_constants::O_TRUNC(); }
            case (none) { }
        }
    }
    auto fd =
        os::libc::open(str::buf(path), fflags,
                       os::libc_constants::S_IRUSR() |
                           os::libc_constants::S_IWUSR());
    if (fd < 0) {
        log_err "error opening file for writing";
        log_err sys::rustrt::last_os_error();
        fail;
    }
    ret fd_buf_writer(fd, true);
}

type writer =
    obj {
        fn get_buf_writer() -> buf_writer ;

            // write_str will continue to do utf-8 output only. an alternative
            // function will be provided for general encoded string output
            fn write_str(str) ;
        fn write_line(str) ;
        fn write_char(char) ;
        fn write_int(int) ;
        fn write_uint(uint) ;
        fn write_bytes(vec[u8]) ;
        fn write_le_uint(uint, uint) ;
        fn write_le_int(int, uint) ;
        fn write_be_uint(uint, uint) ;
    };

fn uint_to_le_bytes(uint n, uint size) -> vec[u8] {
    let vec[u8] bytes = [];
    while (size > 0u) { bytes += [n & 255u as u8]; n >>= 8u; size -= 1u; }
    ret bytes;
}

fn uint_to_be_bytes(uint n, uint size) -> vec[u8] {
    let vec[u8] bytes = [];
    auto i = size - 1u as int;
    while (i >= 0) { bytes += [n >> (i * 8 as uint) & 255u as u8]; i -= 1; }
    ret bytes;
}

obj new_writer(buf_writer out) {
    fn get_buf_writer() -> buf_writer { ret out; }
    fn write_str(str s) { out.write(str::bytes(s)); }
    fn write_line(str s) {
        out.write(str::bytes(s));
        out.write(str::bytes("\n"));
    }
    fn write_char(char ch) {
        // FIXME needlessly consy

        out.write(str::bytes(str::from_char(ch)));
    }
    fn write_int(int n) { out.write(str::bytes(int::to_str(n, 10u))); }
    fn write_uint(uint n) { out.write(str::bytes(uint::to_str(n, 10u))); }
    fn write_bytes(vec[u8] bytes) { out.write(bytes); }
    fn write_le_uint(uint n, uint size) {
        out.write(uint_to_le_bytes(n, size));
    }
    fn write_le_int(int n, uint size) {
        out.write(uint_to_le_bytes(n as uint, size));
    }
    fn write_be_uint(uint n, uint size) {
        out.write(uint_to_be_bytes(n, size));
    }
}


// FIXME: Remove me once objects are exported.
fn new_writer_(buf_writer out) -> writer { ret new_writer(out); }

fn file_writer(str path, vec[fileflag] flags) -> writer {
    ret new_writer(file_buf_writer(path, flags));
}


// FIXME: fileflags
fn buffered_file_buf_writer(str path) -> buf_writer {
    auto f = os::libc::fopen(str::buf(path), str::buf("w"));
    if (f as uint == 0u) { log_err "error opening " + path; fail; }
    ret FILE_writer(f, true);
}


// FIXME it would be great if this could be a const
fn stdout() -> writer { ret new_writer(fd_buf_writer(1, false)); }

type str_writer =
    obj {
        fn get_writer() -> writer ;
        fn get_str() -> str ;
    };

type mutable_byte_buf = @rec(mutable vec[mutable u8] buf, mutable uint pos);

obj byte_buf_writer(mutable_byte_buf buf) {
    fn write(vec[u8] v) {
        // Fast path.

        if (buf.pos == vec::len(buf.buf)) {
            // FIXME: Fix our type system. There's no reason you shouldn't be
            // able to add a mutable vector to an immutable one.

            auto mv = vec::rustrt::unsafe_vec_to_mut[u8](v);
            buf.buf += mv;
            buf.pos += vec::len[u8](v);
            ret;
        }
        // FIXME: Optimize: These should be unique pointers.

        auto vlen = vec::len[u8](v);
        auto vpos = 0u;
        while (vpos < vlen) {
            auto b = v.(vpos);
            if (buf.pos == vec::len(buf.buf)) {
                buf.buf += [mutable b];
            } else { buf.buf.(buf.pos) = b; }
            buf.pos += 1u;
            vpos += 1u;
        }
    }
    fn seek(int offset, seek_style whence) {
        auto pos = buf.pos;
        auto len = vec::len(buf.buf);
        buf.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { ret buf.pos; }
}

fn string_writer() -> str_writer {
    // FIXME: yikes, this is bad. Needs fixing of mutable syntax.

    let vec[mutable u8] b = [mutable 0u8];
    vec::pop(b);
    let mutable_byte_buf buf = @rec(mutable buf=b, mutable pos=0u);
    obj str_writer_wrap(writer wr, mutable_byte_buf buf) {
        fn get_writer() -> writer { ret wr; }
        fn get_str() -> str { ret str::unsafe_from_bytes(buf.buf); }
    }
    ret str_writer_wrap(new_writer(byte_buf_writer(buf)), buf);
}


// Utility functions
fn seek_in_buf(int offset, uint pos, uint len, seek_style whence) -> uint {
    auto bpos = pos as int;
    auto blen = len as int;
    alt (whence) {
        case (seek_set) { bpos = offset; }
        case (seek_cur) { bpos += offset; }
        case (seek_end) { bpos = blen + offset; }
    }
    if (bpos < 0) { bpos = 0; } else if (bpos > blen) { bpos = blen; }
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
