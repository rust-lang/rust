/*
Module: io

Basic input/output
*/

import core::ctypes::fd_t;
import core::ctypes::c_int;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_get_stdin() -> os::libc::FILE;
    fn rust_get_stdout() -> os::libc::FILE;
    fn rust_get_stderr() -> os::libc::FILE;
}

// Reading

// FIXME This is all buffered. We might need an unbuffered variant as well
enum seek_style { seek_set; seek_end; seek_cur; }


// The raw underlying reader iface. All readers must implement this.
iface reader {
    // FIXME: Seekable really should be orthogonal.
    fn read_bytes(uint) -> [u8];
    fn read_byte() -> int;
    fn unread_byte(int);
    fn eof() -> bool;
    fn seek(int, seek_style);
    fn tell() -> uint;
}

// Generic utility functions defined on readers

impl reader_util for reader {
    fn read_chars(n: uint) -> [char] {
        // returns the (consumed offset, n_req), appends characters to &chars
        fn chars_from_buf(buf: [u8], &chars: [char]) -> (uint, uint) {
            let i = 0u;
            while i < vec::len(buf) {
                let b0 = buf[i];
                let w = str::utf8_char_width(b0);
                let end = i + w;
                i += 1u;
                assert (w > 0u);
                if w == 1u {
                    chars += [ b0 as char ];
                    cont;
                }
                // can't satisfy this char with the existing data
                if end > vec::len(buf) {
                    ret (i - 1u, end - vec::len(buf));
                }
                let val = 0u;
                while i < end {
                    let next = buf[i] as int;
                    i += 1u;
                    assert (next > -1);
                    assert (next & 192 == 128);
                    val <<= 6u;
                    val += next & 63 as uint;
                }
                // See str::char_at
                val += (b0 << (w + 1u as u8) as uint)
                    << (w - 1u) * 6u - w - 1u;
                chars += [ val as char ];
            }
            ret (i, 0u);
        }
        let buf: [u8] = [];
        let chars: [char] = [];
        // might need more bytes, but reading n will never over-read
        let nbread = n;
        while nbread > 0u {
            let data = self.read_bytes(nbread);
            if vec::len(data) == 0u {
                // eof - FIXME should we do something if
                // we're split in a unicode char?
                break;
            }
            buf += data;
            let (offset, nbreq) = chars_from_buf(buf, chars);
            let ncreq = n - vec::len(chars);
            // again we either know we need a certain number of bytes
            // to complete a character, or we make sure we don't
            // over-read by reading 1-byte per char needed
            nbread = if ncreq > nbreq { ncreq } else { nbreq };
            if nbread > 0u {
                buf = vec::slice(buf, offset, vec::len(buf));
            }
        }
        chars
    }

    fn read_char() -> char {
        let c = self.read_chars(1u);
        if vec::len(c) == 0u {
            ret -1 as char; // FIXME will this stay valid?
        }
        assert(vec::len(c) == 1u);
        ret c[0];
    }

    fn read_line() -> str {
        let buf: [u8] = [];
        while true {
            let ch = self.read_byte();
            if ch == -1 || ch == 10 { break; }
            buf += [ch as u8];
        }
        str::unsafe_from_bytes(buf)
    }

    fn read_c_str() -> str {
        let buf: [u8] = [];
        while true {
            let ch = self.read_byte();
            if ch < 1 { break; } else { buf += [ch as u8]; }
        }
        str::unsafe_from_bytes(buf)
    }

    // FIXME deal with eof?
    fn read_le_uint(size: uint) -> uint {
        let val = 0u, pos = 0u, i = size;
        while i > 0u {
            val += (self.read_byte() as uint) << pos;
            pos += 8u;
            i -= 1u;
        }
        val
    }
    fn read_le_int(size: uint) -> int {
        let val = 0u, pos = 0u, i = size;
        while i > 0u {
            val += (self.read_byte() as uint) << pos;
            pos += 8u;
            i -= 1u;
        }
        val as int
    }
    fn read_be_uint(size: uint) -> uint {
        let val = 0u, i = size;
        while i > 0u {
            i -= 1u;
            val += (self.read_byte() as uint) << i * 8u;
        }
        val
    }

    fn read_whole_stream() -> [u8] {
        let buf: [u8] = [];
        while !self.eof() { buf += self.read_bytes(2048u); }
        buf
    }
}

// Reader implementations

fn convert_whence(whence: seek_style) -> i32 {
    ret alt whence {
      seek_set { 0i32 }
      seek_cur { 1i32 }
      seek_end { 2i32 }
    };
}

impl of reader for os::libc::FILE {
    fn read_bytes(len: uint) -> [u8] unsafe {
        let buf = [];
        vec::reserve(buf, len);
        let read = os::libc::fread(vec::unsafe::to_ptr(buf), 1u, len, self);
        vec::unsafe::set_len(buf, read);
        ret buf;
    }
    fn read_byte() -> int { ret os::libc::fgetc(self) as int; }
    fn unread_byte(byte: int) { os::libc::ungetc(byte as i32, self); }
    fn eof() -> bool { ret os::libc::feof(self) != 0i32; }
    fn seek(offset: int, whence: seek_style) {
        assert os::libc::fseek(self, offset, convert_whence(whence)) == 0i32;
    }
    fn tell() -> uint { ret os::libc::ftell(self) as uint; }
}

// A forwarding impl of reader that also holds on to a resource for the
// duration of its lifetime.
// FIXME there really should be a better way to do this
impl <T: reader, C> of reader for {base: T, cleanup: C} {
    fn read_bytes(len: uint) -> [u8] { self.base.read_bytes(len) }
    fn read_byte() -> int { self.base.read_byte() }
    fn unread_byte(byte: int) { self.base.unread_byte(byte); }
    fn eof() -> bool { self.base.eof() }
    fn seek(off: int, whence: seek_style) { self.base.seek(off, whence) }
    fn tell() -> uint { self.base.tell() }
}

resource FILE_res(f: os::libc::FILE) { os::libc::fclose(f); }

fn FILE_reader(f: os::libc::FILE, cleanup: bool) -> reader {
    if cleanup {
        {base: f, cleanup: FILE_res(f)} as reader
    } else {
        f as reader
    }
}

// FIXME: this should either be an iface-less impl, a set of top-level
// functions that take a reader, or a set of default methods on reader
// (which can then be called reader)

fn stdin() -> reader { rustrt::rust_get_stdin() as reader }

fn file_reader(path: str) -> result::t<reader, str> {
    let f = str::as_buf(path, {|pathbuf|
        str::as_buf("r", {|modebuf|
            os::libc::fopen(pathbuf, modebuf)
        })
    });
    ret if f as uint == 0u { result::err("error opening " + path) }
    else {
        result::ok(FILE_reader(f, true))
    }
}


// Byte buffer readers

// TODO: const u8, but this fails with rustboot.
type byte_buf = {buf: [u8], mutable pos: uint};

impl of reader for byte_buf {
    fn read_bytes(len: uint) -> [u8] {
        let rest = vec::len(self.buf) - self.pos;
        let to_read = len;
        if rest < to_read { to_read = rest; }
        let range = vec::slice(self.buf, self.pos, self.pos + to_read);
        self.pos += to_read;
        ret range;
    }
    fn read_byte() -> int {
        if self.pos == vec::len(self.buf) { ret -1; }
        let b = self.buf[self.pos];
        self.pos += 1u;
        ret b as int;
    }
    fn unread_byte(_byte: int) { #error("TODO: unread_byte"); fail; }
    fn eof() -> bool { self.pos == vec::len(self.buf) }
    fn seek(offset: int, whence: seek_style) {
        let pos = self.pos;
        let len = vec::len(self.buf);
        self.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { self.pos }
}

fn bytes_reader(bytes: [u8]) -> reader {
    {buf: bytes, mutable pos: 0u} as reader
}

fn string_reader(s: str) -> reader {
    bytes_reader(str::bytes(s))
}


// Writing
enum fileflag { append; create; truncate; none; }

// FIXME: Seekable really should be orthogonal.
// FIXME: eventually u64
iface writer {
    fn write([const u8]);
    fn seek(int, seek_style);
    fn tell() -> uint;
    fn flush() -> int;
}

impl <T: writer, C> of writer for {base: T, cleanup: C} {
    fn write(bs: [const u8]) { self.base.write(bs); }
    fn seek(off: int, style: seek_style) { self.base.seek(off, style); }
    fn tell() -> uint { self.base.tell() }
    fn flush() -> int { self.base.flush() }
}

impl of writer for os::libc::FILE {
    fn write(v: [const u8]) unsafe {
        let len = vec::len(v);
        let vbuf = vec::unsafe::to_ptr(v);
        let nout = os::libc::fwrite(vbuf, len, 1u, self);
        if nout < 1u { #error("error dumping buffer"); }
    }
    fn seek(offset: int, whence: seek_style) {
        assert os::libc::fseek(self, offset, convert_whence(whence)) == 0i32;
    }
    fn tell() -> uint { os::libc::ftell(self) as uint }
    fn flush() -> int { os::libc::fflush(self) as int }
}

fn FILE_writer(f: os::libc::FILE, cleanup: bool) -> writer {
    if cleanup {
        {base: f, cleanup: FILE_res(f)} as writer
    } else {
        f as writer
    }
}

impl of writer for fd_t {
    fn write(v: [const u8]) unsafe {
        let len = vec::len(v);
        let count = 0u;
        let vbuf;
        while count < len {
            vbuf = ptr::offset(vec::unsafe::to_ptr(v), count);
            let nout = os::libc::write(self, vbuf, len);
            if nout < 0 {
                #error("error dumping buffer");
                log(error, sys::last_os_error());
                fail;
            }
            count += nout as uint;
        }
    }
    fn seek(_offset: int, _whence: seek_style) {
        #error("need 64-bit native calls for seek, sorry");
        fail;
    }
    fn tell() -> uint {
        #error("need 64-bit native calls for tell, sorry");
        fail;
    }
    fn flush() -> int { 0 }
}

resource fd_res(fd: fd_t) { os::libc::close(fd); }

fn fd_writer(fd: fd_t, cleanup: bool) -> writer {
    if cleanup {
        {base: fd, cleanup: fd_res(fd)} as writer
    } else {
        fd as writer
    }
}

fn mk_file_writer(path: str, flags: [fileflag])
    -> result::t<writer, str> {
    let fflags: i32 =
        os::libc_constants::O_WRONLY | os::libc_constants::O_BINARY;
    for f: fileflag in flags {
        alt f {
          append { fflags |= os::libc_constants::O_APPEND; }
          create { fflags |= os::libc_constants::O_CREAT; }
          truncate { fflags |= os::libc_constants::O_TRUNC; }
          none { }
        }
    }
    let fd = str::as_buf(path, {|pathbuf|
        os::libc::open(pathbuf, fflags, os::libc_constants::S_IRUSR |
                       os::libc_constants::S_IWUSR)
    });
    if fd < 0i32 {
        // FIXME don't log this! put it in the returned error string
        log(error, sys::last_os_error());
        result::err("error opening " + path)
    } else {
        result::ok(fd_writer(fd, true))
    }
}

fn uint_to_le_bytes(n: uint, size: uint) -> [u8] {
    let bytes: [u8] = [], i = size, n = n;
    while i > 0u { bytes += [n & 255u as u8]; n >>= 8u; i -= 1u; }
    ret bytes;
}

fn uint_to_be_bytes(n: uint, size: uint) -> [u8] {
    let bytes: [u8] = [];
    let i = size - 1u as int;
    while i >= 0 { bytes += [n >> (i * 8 as uint) & 255u as u8]; i -= 1; }
    ret bytes;
}

impl writer_util for writer {
    fn write_char(ch: char) {
        if ch as uint < 128u {
            self.write([ch as u8]);
        } else {
            self.write(str::bytes(str::from_char(ch)));
        }
    }
    fn write_str(s: str) { self.write(str::bytes(s)); }
    fn write_line(s: str) { self.write(str::bytes(s + "\n")); }
    fn write_int(n: int) { self.write(str::bytes(int::to_str(n, 10u))); }
    fn write_uint(n: uint) { self.write(str::bytes(uint::to_str(n, 10u))); }

    fn write_le_uint(n: uint, size: uint) {
        self.write(uint_to_le_bytes(n, size));
    }
    fn write_le_int(n: int, size: uint) {
        self.write(uint_to_le_bytes(n as uint, size));
    }
    fn write_be_uint(n: uint, size: uint) {
        self.write(uint_to_be_bytes(n, size));
    }
}

fn file_writer(path: str, flags: [fileflag]) -> result::t<writer, str> {
    result::chain(mk_file_writer(path, flags), { |w| result::ok(w)})
}


// FIXME: fileflags
fn buffered_file_writer(path: str) -> result::t<writer, str> {
    let f = str::as_buf(path, {|pathbuf|
        str::as_buf("w", {|modebuf| os::libc::fopen(pathbuf, modebuf) })
    });
    ret if f as uint == 0u { result::err("error opening " + path) }
    else { result::ok(FILE_writer(f, true)) }
}

// FIXME it would be great if this could be a const
fn stdout() -> writer { fd_writer(1i32, false) }
fn stderr() -> writer { fd_writer(2i32, false) }

fn print(s: str) { stdout().write_str(s); }
fn println(s: str) { stdout().write_line(s); }

type mem_buffer = @{mutable buf: [mutable u8],
                    mutable pos: uint};

impl of writer for mem_buffer {
    fn write(v: [const u8]) {
        // Fast path.
        if self.pos == vec::len(self.buf) {
            for b: u8 in v { self.buf += [mutable b]; }
            self.pos += vec::len(v);
            ret;
        }
        // FIXME: Optimize: These should be unique pointers.
        let vlen = vec::len(v);
        let vpos = 0u;
        while vpos < vlen {
            let b = v[vpos];
            if self.pos == vec::len(self.buf) {
                self.buf += [mutable b];
            } else { self.buf[self.pos] = b; }
            self.pos += 1u;
            vpos += 1u;
        }
    }
    fn seek(offset: int, whence: seek_style) {
        let pos = self.pos;
        let len = vec::len(self.buf);
        self.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { self.pos }
    fn flush() -> int { 0 }
}

fn mk_mem_buffer() -> mem_buffer {
    @{mutable buf: [mutable], mutable pos: 0u}
}
fn mem_buffer_writer(b: mem_buffer) -> writer { b as writer }
fn mem_buffer_buf(b: mem_buffer) -> [u8] { vec::from_mut(b.buf) }
fn mem_buffer_str(b: mem_buffer) -> str { str::unsafe_from_bytes(b.buf) }

// Utility functions
fn seek_in_buf(offset: int, pos: uint, len: uint, whence: seek_style) ->
   uint {
    let bpos = pos as int;
    let blen = len as int;
    alt whence {
      seek_set { bpos = offset; }
      seek_cur { bpos += offset; }
      seek_end { bpos = blen + offset; }
    }
    if bpos < 0 { bpos = 0; } else if bpos > blen { bpos = blen; }
    ret bpos as uint;
}

fn read_whole_file_str(file: str) -> result::t<str, str> {
    result::chain(read_whole_file(file), { |bytes|
        result::ok(str::unsafe_from_bytes(bytes))
    })
}

// FIXME implement this in a low-level way. Going through the abstractions is
// pointless.
fn read_whole_file(file: str) -> result::t<[u8], str> {
    result::chain(file_reader(file), { |rdr|
        result::ok(rdr.read_whole_stream())
    })
}

// fsync related

mod fsync {

    enum level {
        // whatever fsync does on that platform
        fsync;

        // fdatasync on linux, similiar or more on other platforms
        fdatasync;

        // full fsync
        //
        // You must additionally sync the parent directory as well!
        fullfsync;
    }


    // Resource of artifacts that need to fsync on destruction
    resource res<t>(arg: arg<t>) {
        alt arg.opt_level {
          option::none { }
          option::some(level) {
            // fail hard if not succesful
            assert(arg.fsync_fn(arg.val, level) != -1);
          }
        }
    }

    type arg<t> = {
        val: t,
        opt_level: option::t<level>,
        fsync_fn: fn@(t, level) -> int
    };

    // fsync file after executing blk
    // FIXME find better way to create resources within lifetime of outer res
    fn FILE_res_sync(&&file: FILE_res, opt_level: option::t<level>,
                  blk: block(&&res<os::libc::FILE>)) {
        blk(res({
            val: *file, opt_level: opt_level,
            fsync_fn: fn@(&&file: os::libc::FILE, l: level) -> int {
                ret os::fsync_fd(os::libc::fileno(file), l) as int;
            }
        }));
    }

    // fsync fd after executing blk
    fn fd_res_sync(&&fd: fd_res, opt_level: option::t<level>,
                   blk: block(&&res<fd_t>)) {
        blk(res({
            val: *fd, opt_level: opt_level,
            fsync_fn: fn@(&&fd: fd_t, l: level) -> int {
                ret os::fsync_fd(fd, l) as int;
            }
        }));
    }

    // Type of objects that may want to fsync
    iface t { fn fsync(l: level) -> int; }

    // Call o.fsync after executing blk
    fn obj_sync(&&o: t, opt_level: option::t<level>, blk: block(&&res<t>)) {
        blk(res({
            val: o, opt_level: opt_level,
            fsync_fn: fn@(&&o: t, l: level) -> int { ret o.fsync(l); }
        }));
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_simple() {
        let tmpfile: str = "tmp/lib-io-test-simple.tmp";
        log(debug, tmpfile);
        let frood: str = "A hoopy frood who really knows where his towel is.";
        log(debug, frood);
        {
            let out: io::writer =
                result::get(
                    io::file_writer(tmpfile, [io::create, io::truncate]));
            out.write_str(frood);
        }
        let inp: io::reader = result::get(io::file_reader(tmpfile));
        let frood2: str = inp.read_c_str();
        log(debug, frood2);
        assert (str::eq(frood, frood2));
    }

    #[test]
    fn test_readchars_empty() {
        let inp : io::reader = io::string_reader("");
        let res : [char] = inp.read_chars(128u);
        assert(vec::len(res) == 0u);
    }

    #[test]
    fn test_readchars_wide() {
        let wide_test = "生锈的汤匙切肉汤hello生锈的汤匙切肉汤";
        let ivals : [int] = [
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748,
            104, 101, 108, 108, 111,
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748];
        fn check_read_ln(len : uint, s: str, ivals: [int]) {
            let inp : io::reader = io::string_reader(s);
            let res : [char] = inp.read_chars(len);
            if (len <= vec::len(ivals)) {
                assert(vec::len(res) == len);
            }
            assert(vec::slice(ivals, 0u, vec::len(res)) ==
                   vec::map(res, {|x| x as int}));
        }
        let i = 0u;
        while i < 8u {
            check_read_ln(i, wide_test, ivals);
            i += 1u;
        }
        // check a long read for good measure
        check_read_ln(128u, wide_test, ivals);
    }

    #[test]
    fn test_readchar() {
        let inp : io::reader = io::string_reader("生");
        let res : char = inp.read_char();
        assert(res as int == 29983);
    }

    #[test]
    fn test_readchar_empty() {
        let inp : io::reader = io::string_reader("");
        let res : char = inp.read_char();
        assert(res as int == -1);
    }

    #[test]
    fn file_reader_not_exist() {
        alt io::file_reader("not a file") {
          result::err(e) {
            assert e == "error opening not a file";
          }
          result::ok(_) { fail; }
        }
    }

    #[test]
    fn file_writer_bad_name() {
        alt io::file_writer("?/?", []) {
          result::err(e) {
            assert e == "error opening ?/?";
          }
          result::ok(_) { fail; }
        }
    }

    #[test]
    fn buffered_file_writer_bad_name() {
        alt io::buffered_file_writer("?/?") {
          result::err(e) {
            assert e == "error opening ?/?";
          }
          result::ok(_) { fail; }
        }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
