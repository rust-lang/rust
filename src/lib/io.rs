
import os::libc;

native "rust" mod rustrt {
    fn rust_get_stdin() -> os::libc::FILE;
    fn rust_get_stdout() -> os::libc::FILE;
    fn rust_get_stderr() -> os::libc::FILE;
}


// Reading

// FIXME This is all buffered. We might need an unbuffered variant as well
tag seek_style { seek_set; seek_end; seek_cur; }


// The raw underlying reader class. All readers must implement this.
type buf_reader =
    // FIXME: Seekable really should be orthogonal. We will need
    // inheritance.
    obj {
        fn read(uint) -> [u8];
        fn read_byte() -> int;
        fn unread_byte(int);
        fn eof() -> bool;
        fn seek(int, seek_style);
        fn tell() -> uint;
    };


// Convenience methods for reading.
type reader =
    // FIXME: This should inherit from buf_reader.
    // FIXME: eventually u64

    obj {
        fn get_buf_reader() -> buf_reader;
        fn read_byte() -> int;
        fn unread_byte(int);
        fn read_bytes(uint) -> [u8];
        fn read_char() -> char;
        fn eof() -> bool;
        fn read_line() -> str;
        fn read_c_str() -> str;
        fn read_le_uint(uint) -> uint;
        fn read_le_int(uint) -> int;
        fn read_be_uint(uint) -> uint;
        fn read_whole_stream() -> [u8];
        fn seek(int, seek_style);
        fn tell() -> uint;
    };

fn convert_whence(whence: seek_style) -> int {
    ret alt whence { seek_set. { 0 } seek_cur. { 1 } seek_end. { 2 } };
}

resource FILE_res(f: os::libc::FILE) { os::libc::fclose(f); }

obj FILE_buf_reader(f: os::libc::FILE, res: option::t<@FILE_res>) {
    fn read(len: uint) -> [u8] unsafe {
        let buf = [];
        vec::reserve::<u8>(buf, len);
        let read =
            os::libc::fread(vec::unsafe::to_ptr::<u8>(buf), 1u, len, f);
        vec::unsafe::set_len::<u8>(buf, read);
        ret buf;
    }
    fn read_byte() -> int { ret os::libc::fgetc(f); }
    fn unread_byte(byte: int) { os::libc::ungetc(byte, f); }
    fn eof() -> bool { ret os::libc::feof(f) != 0; }
    fn seek(offset: int, whence: seek_style) {
        assert (os::libc::fseek(f, offset, convert_whence(whence)) == 0);
    }
    fn tell() -> uint { ret os::libc::ftell(f) as uint; }
}


// FIXME: Convert this into pseudomethods on buf_reader.
obj new_reader(rdr: buf_reader) {
    fn get_buf_reader() -> buf_reader { ret rdr; }
    fn read_byte() -> int { ret rdr.read_byte(); }
    fn unread_byte(byte: int) { ret rdr.unread_byte(byte); }
    fn read_bytes(len: uint) -> [u8] { ret rdr.read(len); }
    fn read_char() -> char {
        let c0 = rdr.read_byte();
        if c0 == -1 {
            ret -1 as char; // FIXME will this stay valid?

        }
        let b0 = c0 as u8;
        let w = str::utf8_char_width(b0);
        assert (w > 0u);
        if w == 1u { ret b0 as char; }
        let val = 0u;
        while w > 1u {
            w -= 1u;
            let next = rdr.read_byte();
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
        let buf: [u8] = [];
        // No break yet in rustc

        let go_on = true;
        while go_on {
            let ch = rdr.read_byte();
            if ch == -1 || ch == 10 {
                go_on = false;
            } else { buf += [ch as u8]; }
        }
        ret str::unsafe_from_bytes(buf);
    }
    fn read_c_str() -> str {
        let buf: [u8] = [];
        let go_on = true;
        while go_on {
            let ch = rdr.read_byte();
            if ch < 1 { go_on = false; } else { buf += [ch as u8]; }
        }
        ret str::unsafe_from_bytes(buf);
    }

    // FIXME deal with eof?
    fn read_le_uint(size: uint) -> uint {
        let val = 0u;
        let pos = 0u;
        while size > 0u {
            val += (rdr.read_byte() as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val;
    }
    fn read_le_int(size: uint) -> int {
        let val = 0u;
        let pos = 0u;
        while size > 0u {
            val += (rdr.read_byte() as uint) << pos;
            pos += 8u;
            size -= 1u;
        }
        ret val as int;
    }

    // FIXME deal with eof?
    fn read_be_uint(sz: uint) -> uint {
        let val = 0u;

        while sz > 0u {
            sz -= 1u;
            val += (rdr.read_byte() as uint) << sz * 8u;
        }
        ret val;
    }
    fn read_whole_stream() -> [u8] {
        let buf: [u8] = [];
        while !rdr.eof() { buf += rdr.read(2048u); }
        ret buf;
    }
    fn seek(offset: int, whence: seek_style) { ret rdr.seek(offset, whence); }
    fn tell() -> uint { ret rdr.tell(); }
}

fn stdin() -> reader {
    ret new_reader(FILE_buf_reader(rustrt::rust_get_stdin(), option::none));
}

fn file_reader(path: str) -> reader {
    let f = str::as_buf(path, {|pathbuf|
        str::as_buf("r", {|modebuf|
            os::libc::fopen(pathbuf, modebuf)
        })
    });
    if f as uint == 0u { log_err "error opening " + path; fail; }
    ret new_reader(FILE_buf_reader(f, option::some(@FILE_res(f))));
}


// Byte buffer readers

// TODO: mutable? u8, but this fails with rustboot.
type byte_buf = @{buf: [u8], mutable pos: uint};

obj byte_buf_reader(bbuf: byte_buf) {
    fn read(len: uint) -> [u8] {
        let rest = vec::len::<u8>(bbuf.buf) - bbuf.pos;
        let to_read = len;
        if rest < to_read { to_read = rest; }
        let range = vec::slice::<u8>(bbuf.buf, bbuf.pos, bbuf.pos + to_read);
        bbuf.pos += to_read;
        ret range;
    }
    fn read_byte() -> int {
        if bbuf.pos == vec::len::<u8>(bbuf.buf) { ret -1; }
        let b = bbuf.buf[bbuf.pos];
        bbuf.pos += 1u;
        ret b as int;
    }
    fn unread_byte(_byte: int) { log_err "TODO: unread_byte"; fail; }
    fn eof() -> bool { ret bbuf.pos == vec::len::<u8>(bbuf.buf); }
    fn seek(offset: int, whence: seek_style) {
        let pos = bbuf.pos;
        let len = vec::len::<u8>(bbuf.buf);
        bbuf.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { ret bbuf.pos; }
}

fn new_byte_buf_reader(buf: [u8]) -> buf_reader {
    ret byte_buf_reader(@{buf: buf, mutable pos: 0u});
}

fn string_reader(s: str) -> reader {
    ret new_reader(new_byte_buf_reader(str::bytes(s)));
}


// Writing
tag fileflag { append; create; truncate; none; }

type buf_writer =
    // FIXME: Seekable really should be orthogonal. We will need
    // inheritance.
    // FIXME: eventually u64

    obj {
        fn write([u8]);
        fn seek(int, seek_style);
        fn tell() -> uint;
    };

obj FILE_writer(f: os::libc::FILE, res: option::t<@FILE_res>) {
    fn write(v: [u8]) unsafe {
        let len = vec::len::<u8>(v);
        let vbuf = vec::unsafe::to_ptr::<u8>(v);
        let nout = os::libc::fwrite(vbuf, len, 1u, f);
        if nout < 1u { log_err "error dumping buffer"; }
    }
    fn seek(offset: int, whence: seek_style) {
        assert (os::libc::fseek(f, offset, convert_whence(whence)) == 0);
    }
    fn tell() -> uint { ret os::libc::ftell(f) as uint; }
}

resource fd_res(fd: int) { os::libc::close(fd); }

obj fd_buf_writer(fd: int, res: option::t<@fd_res>) {
    fn write(v: [u8]) unsafe {
        let len = vec::len::<u8>(v);
        let count = 0u;
        let vbuf;
        while count < len {
            vbuf = ptr::offset(vec::unsafe::to_ptr::<u8>(v), count);
            let nout = os::libc::write(fd, vbuf, len);
            if nout < 0 {
                log_err "error dumping buffer";
                log_err sys::last_os_error();
                fail;
            }
            count += nout as uint;
        }
    }
    fn seek(_offset: int, _whence: seek_style) {
        log_err "need 64-bit native calls for seek, sorry";
        fail;
    }
    fn tell() -> uint {
        log_err "need 64-bit native calls for tell, sorry";
        fail;
    }
}

fn file_buf_writer(path: str, flags: [fileflag]) -> buf_writer {
    let fflags: int =
        os::libc_constants::O_WRONLY() | os::libc_constants::O_BINARY();
    for f: fileflag in flags {
        alt f {
          append. { fflags |= os::libc_constants::O_APPEND(); }
          create. { fflags |= os::libc_constants::O_CREAT(); }
          truncate. { fflags |= os::libc_constants::O_TRUNC(); }
          none. { }
        }
    }
    let fd =
        str::as_buf(path,
                    {|pathbuf|
                        os::libc::open(pathbuf, fflags,
                                       os::libc_constants::S_IRUSR() |
                                           os::libc_constants::S_IWUSR())
                    });
    if fd < 0 {
        log_err "error opening file for writing";
        log_err sys::last_os_error();
        fail;
    }
    ret fd_buf_writer(fd, option::some(@fd_res(fd)));
}

type writer =
    // write_str will continue to do utf-8 output only. an alternative
    // function will be provided for general encoded string output
    obj {
        fn get_buf_writer() -> buf_writer;
        fn write_str(str);
        fn write_line(str);
        fn write_char(char);
        fn write_int(int);
        fn write_uint(uint);
        fn write_bytes([u8]);
        fn write_le_uint(uint, uint);
        fn write_le_int(int, uint);
        fn write_be_uint(uint, uint);
    };

fn uint_to_le_bytes(n: uint, size: uint) -> [u8] {
    let bytes: [u8] = [];
    while size > 0u { bytes += [n & 255u as u8]; n >>= 8u; size -= 1u; }
    ret bytes;
}

fn uint_to_be_bytes(n: uint, size: uint) -> [u8] {
    let bytes: [u8] = [];
    let i = size - 1u as int;
    while i >= 0 { bytes += [n >> (i * 8 as uint) & 255u as u8]; i -= 1; }
    ret bytes;
}

obj new_writer(out: buf_writer) {
    fn get_buf_writer() -> buf_writer { ret out; }
    fn write_str(s: str) { out.write(str::bytes(s)); }
    fn write_line(s: str) {
        out.write(str::bytes(s));
        out.write(str::bytes("\n"));
    }
    fn write_char(ch: char) {
        // FIXME needlessly consy

        out.write(str::bytes(str::from_char(ch)));
    }
    fn write_int(n: int) { out.write(str::bytes(int::to_str(n, 10u))); }
    fn write_uint(n: uint) { out.write(str::bytes(uint::to_str(n, 10u))); }
    fn write_bytes(bytes: [u8]) { out.write(bytes); }
    fn write_le_uint(n: uint, size: uint) {
        out.write(uint_to_le_bytes(n, size));
    }
    fn write_le_int(n: int, size: uint) {
        out.write(uint_to_le_bytes(n as uint, size));
    }
    fn write_be_uint(n: uint, size: uint) {
        out.write(uint_to_be_bytes(n, size));
    }
}

fn file_writer(path: str, flags: [fileflag]) -> writer {
    ret new_writer(file_buf_writer(path, flags));
}


// FIXME: fileflags
fn buffered_file_buf_writer(path: str) -> buf_writer {
    let f =
        str::as_buf(path,
                    {|pathbuf|
                        str::as_buf("w",
                                    {|modebuf|
                                        os::libc::fopen(pathbuf, modebuf)
                                    })
                    });
    if f as uint == 0u { log_err "error opening " + path; fail; }
    ret FILE_writer(f, option::some(@FILE_res(f)));
}


// FIXME it would be great if this could be a const
fn stdout() -> writer { ret new_writer(fd_buf_writer(1, option::none)); }
fn stderr() -> writer { ret new_writer(fd_buf_writer(2, option::none)); }

fn print(s: str) { stdout().write_str(s); }
fn println(s: str) { stdout().write_str(s + "\n"); }

type str_writer =
    obj {
        fn get_writer() -> writer;
        fn get_str() -> str;
    };

type mutable_byte_buf = @{mutable buf: [mutable u8], mutable pos: uint};

obj byte_buf_writer(buf: mutable_byte_buf) {
    fn write(v: [u8]) {
        // Fast path.

        if buf.pos == vec::len(buf.buf) {
            for b: u8 in v { buf.buf += [mutable b]; }
            buf.pos += vec::len::<u8>(v);
            ret;
        }
        // FIXME: Optimize: These should be unique pointers.

        let vlen = vec::len::<u8>(v);
        let vpos = 0u;
        while vpos < vlen {
            let b = v[vpos];
            if buf.pos == vec::len(buf.buf) {
                buf.buf += [mutable b];
            } else { buf.buf[buf.pos] = b; }
            buf.pos += 1u;
            vpos += 1u;
        }
    }
    fn seek(offset: int, whence: seek_style) {
        let pos = buf.pos;
        let len = vec::len(buf.buf);
        buf.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { ret buf.pos; }
}

fn string_writer() -> str_writer {
    // FIXME: yikes, this is bad. Needs fixing of mutable syntax.

    let b: [mutable u8] = [mutable 0u8];
    vec::pop(b);
    let buf: mutable_byte_buf = @{mutable buf: b, mutable pos: 0u};
    obj str_writer_wrap(wr: writer, buf: mutable_byte_buf) {
        fn get_writer() -> writer { ret wr; }
        fn get_str() -> str { ret str::unsafe_from_bytes(buf.buf); }
    }
    ret str_writer_wrap(new_writer(byte_buf_writer(buf)), buf);
}


// Utility functions
fn seek_in_buf(offset: int, pos: uint, len: uint, whence: seek_style) ->
   uint {
    let bpos = pos as int;
    let blen = len as int;
    alt whence {
      seek_set. { bpos = offset; }
      seek_cur. { bpos += offset; }
      seek_end. { bpos = blen + offset; }
    }
    if bpos < 0 { bpos = 0; } else if bpos > blen { bpos = blen; }
    ret bpos as uint;
}

fn read_whole_file_str(file: str) -> str {
    str::unsafe_from_bytes(read_whole_file(file))
}

fn read_whole_file(file: str) -> [u8] {

    // FIXME: There's a lot of copying here
    file_reader(file).read_whole_stream()
}



//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
