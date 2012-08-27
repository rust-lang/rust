/*
Module: io

Basic input/output
*/

import result::Result;

import cmp::Eq;
import dvec::DVec;
import libc::{c_int, c_long, c_uint, c_void, size_t, ssize_t};
import libc::consts::os::posix88::*;
import libc::consts::os::extra::*;

#[allow(non_camel_case_types)] // not sure what to do about this
type fd_t = c_int;

#[abi = "cdecl"]
extern mod rustrt {
    fn rust_get_stdin() -> *libc::FILE;
    fn rust_get_stdout() -> *libc::FILE;
    fn rust_get_stderr() -> *libc::FILE;
}

// Reading

// FIXME (#2004): This is all buffered. We might need an unbuffered variant
// as well
enum SeekStyle { SeekSet, SeekEnd, SeekCur, }


// The raw underlying reader trait. All readers must implement this.
trait Reader {
    // FIXME (#2004): Seekable really should be orthogonal.

    // FIXME (#2982): This should probably return an error.
    fn read(buf: &[mut u8], len: uint) -> uint;
    fn read_byte() -> int;
    fn unread_byte(int);
    fn eof() -> bool;
    fn seek(int, SeekStyle);
    fn tell() -> uint;
}

// Generic utility functions defined on readers

trait ReaderUtil {
    fn read_bytes(len: uint) -> ~[u8];
    fn read_line() -> ~str;
}

impl<T: Reader> T : ReaderUtil {
    fn read_bytes(len: uint) -> ~[u8] {
        let mut buf = ~[mut];
        vec::reserve(buf, len);
        unsafe { vec::unsafe::set_len(buf, len); }

        let count = self.read(buf, len);

        unsafe { vec::unsafe::set_len(buf, count); }
        vec::from_mut(buf)
    }
    fn read_line() -> ~str {
        let mut buf = ~[];
        loop {
            let ch = self.read_byte();
            if ch == -1 || ch == 10 { break; }
            vec::push(buf, ch as u8);
        }
        str::from_bytes(buf)
    }
}

impl Reader {
    fn read_chars(n: uint) -> ~[char] {
        // returns the (consumed offset, n_req), appends characters to &chars
        fn chars_from_buf(buf: ~[u8], &chars: ~[char]) -> (uint, uint) {
            let mut i = 0u;
            while i < vec::len(buf) {
                let b0 = buf[i];
                let w = str::utf8_char_width(b0);
                let end = i + w;
                i += 1u;
                assert (w > 0u);
                if w == 1u {
                    vec::push(chars,  b0 as char );
                    again;
                }
                // can't satisfy this char with the existing data
                if end > vec::len(buf) {
                    return (i - 1u, end - vec::len(buf));
                }
                let mut val = 0u;
                while i < end {
                    let next = buf[i] as int;
                    i += 1u;
                    assert (next > -1);
                    assert (next & 192 == 128);
                    val <<= 6u;
                    val += (next & 63) as uint;
                }
                // See str::char_at
                val += ((b0 << ((w + 1u) as u8)) as uint)
                    << (w - 1u) * 6u - w - 1u;
                vec::push(chars,  val as char );
            }
            return (i, 0u);
        }
        let mut buf: ~[u8] = ~[];
        let mut chars: ~[char] = ~[];
        // might need more bytes, but reading n will never over-read
        let mut nbread = n;
        while nbread > 0u {
            let data = self.read_bytes(nbread);
            if vec::len(data) == 0u {
                // eof - FIXME (#2004): should we do something if
                // we're split in a unicode char?
                break;
            }
            vec::push_all(buf, data);
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
            return -1 as char; // FIXME will this stay valid? // #2004
        }
        assert(vec::len(c) == 1u);
        return c[0];
    }

    fn read_c_str() -> ~str {
        let mut buf: ~[u8] = ~[];
        loop {
            let ch = self.read_byte();
            if ch < 1 { break; } else { vec::push(buf, ch as u8); }
        }
        str::from_bytes(buf)
    }

    // FIXME deal with eof? // #2004
    fn read_le_uint(size: uint) -> uint {
        let mut val = 0u, pos = 0u, i = size;
        while i > 0u {
            val += (self.read_byte() as uint) << pos;
            pos += 8u;
            i -= 1u;
        }
        val
    }
    fn read_le_int(size: uint) -> int {
        let mut val = 0u, pos = 0u, i = size;
        while i > 0u {
            val += (self.read_byte() as uint) << pos;
            pos += 8u;
            i -= 1u;
        }
        val as int
    }
    fn read_be_uint(size: uint) -> uint {
        let mut val = 0u, i = size;
        while i > 0u {
            i -= 1u;
            val += (self.read_byte() as uint) << i * 8u;
        }
        val
    }

    fn read_whole_stream() -> ~[u8] {
        let mut buf: ~[u8] = ~[];
        while !self.eof() { vec::push_all(buf, self.read_bytes(2048u)); }
        buf
    }

    fn each_byte(it: fn(int) -> bool) {
        while !self.eof() {
            if !it(self.read_byte()) { break; }
        }
    }

    fn each_char(it: fn(char) -> bool) {
        while !self.eof() {
            if !it(self.read_char()) { break; }
        }
    }

    fn each_line(it: fn(~str) -> bool) {
        while !self.eof() {
            if !it(self.read_line()) { break; }
        }
    }
}

// Reader implementations

fn convert_whence(whence: SeekStyle) -> i32 {
    return match whence {
      SeekSet => 0i32,
      SeekCur => 1i32,
      SeekEnd => 2i32
    };
}

impl *libc::FILE: Reader {
    fn read(buf: &[mut u8], len: uint) -> uint {
        do vec::as_buf(buf) |buf_p, buf_len| {
            assert buf_len <= len;

            let count = libc::fread(buf_p as *mut c_void, 1u as size_t,
                                    len as size_t, self);

            count as uint
        }
    }
    fn read_byte() -> int { return libc::fgetc(self) as int; }
    fn unread_byte(byte: int) { libc::ungetc(byte as c_int, self); }
    fn eof() -> bool { return libc::feof(self) != 0 as c_int; }
    fn seek(offset: int, whence: SeekStyle) {
        assert libc::fseek(self, offset as c_long, convert_whence(whence))
            == 0 as c_int;
    }
    fn tell() -> uint { return libc::ftell(self) as uint; }
}

// A forwarding impl of reader that also holds on to a resource for the
// duration of its lifetime.
// FIXME there really should be a better way to do this // #2004
impl<T: Reader, C> {base: T, cleanup: C}: Reader {
    fn read(buf: &[mut u8], len: uint) -> uint { self.base.read(buf, len) }
    fn read_byte() -> int { self.base.read_byte() }
    fn unread_byte(byte: int) { self.base.unread_byte(byte); }
    fn eof() -> bool { self.base.eof() }
    fn seek(off: int, whence: SeekStyle) { self.base.seek(off, whence) }
    fn tell() -> uint { self.base.tell() }
}

struct FILERes {
    let f: *libc::FILE;
    new(f: *libc::FILE) { self.f = f; }
    drop { libc::fclose(self.f); }
}

fn FILE_reader(f: *libc::FILE, cleanup: bool) -> Reader {
    if cleanup {
        {base: f, cleanup: FILERes(f)} as Reader
    } else {
        f as Reader
    }
}

// FIXME (#2004): this should either be an trait-less impl, a set of
// top-level functions that take a reader, or a set of default methods on
// reader (which can then be called reader)

fn stdin() -> Reader { rustrt::rust_get_stdin() as Reader }

fn file_reader(path: &Path) -> Result<Reader, ~str> {
    let f = os::as_c_charp(path.to_str(), |pathbuf| {
        os::as_c_charp("r", |modebuf|
            libc::fopen(pathbuf, modebuf)
        )
    });
    return if f as uint == 0u { result::Err(~"error opening "
                                            + path.to_str()) }
    else {
        result::Ok(FILE_reader(f, true))
    }
}


// Byte buffer readers

type ByteBuf = {buf: &[const u8], mut pos: uint};

impl ByteBuf: Reader {
    fn read(buf: &[mut u8], len: uint) -> uint {
        let count = uint::min(&len, &(self.buf.len() - self.pos));

        vec::u8::memcpy(buf,
                        vec::const_view(self.buf, self.pos, self.buf.len()),
                        count);

        self.pos += count;

        count
    }
    fn read_byte() -> int {
        if self.pos == self.buf.len() { return -1; }
        let b = self.buf[self.pos];
        self.pos += 1u;
        return b as int;
    }
    // FIXME (#2738): implement this
    fn unread_byte(_byte: int) { error!("Unimplemented: unread_byte"); fail; }
    fn eof() -> bool { self.pos == self.buf.len() }
    fn seek(offset: int, whence: SeekStyle) {
        let pos = self.pos;
        self.pos = seek_in_buf(offset, pos, self.buf.len(), whence);
    }
    fn tell() -> uint { self.pos }
}

fn with_bytes_reader<t>(bytes: &[u8], f: fn(Reader) -> t) -> t {
    f({buf: bytes, mut pos: 0u} as Reader)
}

fn with_str_reader<T>(s: &str, f: fn(Reader) -> T) -> T {
    str::byte_slice(s, |bytes| with_bytes_reader(bytes, f))
}

// Writing
enum FileFlag { Append, Create, Truncate, NoFlag, }

// What type of writer are we?
enum WriterType { Screen, File }

impl WriterType: Eq {
    pure fn eq(&&other: WriterType) -> bool {
        match (self, other) {
            (Screen, Screen) | (File, File) => true,
            (Screen, _) | (File, _) => false
        }
    }
}

// FIXME (#2004): Seekable really should be orthogonal.
// FIXME (#2004): eventually u64
trait Writer {
    fn write(v: &[const u8]);
    fn seek(int, SeekStyle);
    fn tell() -> uint;
    fn flush() -> int;
    fn get_type() -> WriterType;
}

impl<T: Writer, C> {base: T, cleanup: C}: Writer {
    fn write(bs: &[const u8]) { self.base.write(bs); }
    fn seek(off: int, style: SeekStyle) { self.base.seek(off, style); }
    fn tell() -> uint { self.base.tell() }
    fn flush() -> int { self.base.flush() }
    fn get_type() -> WriterType { File }
}

impl *libc::FILE: Writer {
    fn write(v: &[const u8]) {
        do vec::as_const_buf(v) |vbuf, len| {
            let nout = libc::fwrite(vbuf as *c_void, len as size_t,
                                    1u as size_t, self);
            if nout < 1 as size_t {
                error!("error writing buffer");
                log(error, os::last_os_error());
                fail;
            }
        }
    }
    fn seek(offset: int, whence: SeekStyle) {
        assert libc::fseek(self, offset as c_long, convert_whence(whence))
            == 0 as c_int;
    }
    fn tell() -> uint { libc::ftell(self) as uint }
    fn flush() -> int { libc::fflush(self) as int }
    fn get_type() -> WriterType {
        let fd = libc::fileno(self);
        if libc::isatty(fd) == 0 { File   }
        else                     { Screen }
    }
}

fn FILE_writer(f: *libc::FILE, cleanup: bool) -> Writer {
    if cleanup {
        {base: f, cleanup: FILERes(f)} as Writer
    } else {
        f as Writer
    }
}

impl fd_t: Writer {
    fn write(v: &[const u8]) {
        let mut count = 0u;
        do vec::as_const_buf(v) |vbuf, len| {
            while count < len {
                let vb = ptr::const_offset(vbuf, count) as *c_void;
                let nout = libc::write(self, vb, len as size_t);
                if nout < 0 as ssize_t {
                    error!("error writing buffer");
                    log(error, os::last_os_error());
                    fail;
                }
                count += nout as uint;
            }
        }
    }
    fn seek(_offset: int, _whence: SeekStyle) {
        error!("need 64-bit foreign calls for seek, sorry");
        fail;
    }
    fn tell() -> uint {
        error!("need 64-bit foreign calls for tell, sorry");
        fail;
    }
    fn flush() -> int { 0 }
    fn get_type() -> WriterType {
        if libc::isatty(self) == 0 { File } else { Screen }
    }
}

struct FdRes {
    let fd: fd_t;
    new(fd: fd_t) { self.fd = fd; }
    drop { libc::close(self.fd); }
}

fn fd_writer(fd: fd_t, cleanup: bool) -> Writer {
    if cleanup {
        {base: fd, cleanup: FdRes(fd)} as Writer
    } else {
        fd as Writer
    }
}


fn mk_file_writer(path: &Path, flags: ~[FileFlag])
    -> Result<Writer, ~str> {

    #[cfg(windows)]
    fn wb() -> c_int { (O_WRONLY | O_BINARY) as c_int }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
    for vec::each(flags) |f| {
        match f {
          Append => fflags |= O_APPEND as c_int,
          Create => fflags |= O_CREAT as c_int,
          Truncate => fflags |= O_TRUNC as c_int,
          NoFlag => ()
        }
    }
    let fd = do os::as_c_charp(path.to_str()) |pathbuf| {
        libc::open(pathbuf, fflags,
                   (S_IRUSR | S_IWUSR) as c_int)
    };
    if fd < (0 as c_int) {
        result::Err(fmt!("error opening %s: %s", path.to_str(),
                         os::last_os_error()))
    } else {
        result::Ok(fd_writer(fd, true))
    }
}

fn u64_to_le_bytes<T>(n: u64, size: uint, f: fn(v: &[u8]) -> T) -> T {
    assert size <= 8u;
    match size {
      1u => f(&[n as u8]),
      2u => f(&[n as u8,
              (n >> 8) as u8]),
      4u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8]),
      8u => f(&[n as u8,
              (n >> 8) as u8,
              (n >> 16) as u8,
              (n >> 24) as u8,
              (n >> 32) as u8,
              (n >> 40) as u8,
              (n >> 48) as u8,
              (n >> 56) as u8]),
      _ => {

        let mut bytes: ~[u8] = ~[], i = size, n = n;
        while i > 0u {
            vec::push(bytes, (n & 255_u64) as u8);
            n >>= 8_u64;
            i -= 1u;
        }
        f(bytes)
      }
    }
}

fn u64_to_be_bytes<T>(n: u64, size: uint, f: fn(v: &[u8]) -> T) -> T {
    assert size <= 8u;
    match size {
      1u => f(&[n as u8]),
      2u => f(&[(n >> 8) as u8,
              n as u8]),
      4u => f(&[(n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      8u => f(&[(n >> 56) as u8,
              (n >> 48) as u8,
              (n >> 40) as u8,
              (n >> 32) as u8,
              (n >> 24) as u8,
              (n >> 16) as u8,
              (n >> 8) as u8,
              n as u8]),
      _ => {
        let mut bytes: ~[u8] = ~[];
        let mut i = size;
        while i > 0u {
            let shift = ((i - 1u) * 8u) as u64;
            vec::push(bytes, (n >> shift) as u8);
            i -= 1u;
        }
        f(bytes)
      }
    }
}

fn u64_from_be_bytes(data: &[const u8], start: uint, size: uint) -> u64 {
    let mut sz = size;
    assert (sz <= 8u);
    let mut val = 0_u64;
    let mut pos = start;
    while sz > 0u {
        sz -= 1u;
        val += (data[pos] as u64) << ((sz * 8u) as u64);
        pos += 1u;
    }
    return val;
}

// FIXME: #3048 combine trait+impl (or just move these to
// default methods on writer)
trait WriterUtil {
    fn write_char(ch: char);
    fn write_str(s: &str);
    fn write_line(s: &str);
    fn write_int(n: int);
    fn write_uint(n: uint);
    fn write_le_uint(n: uint);
    fn write_le_int(n: int);
    fn write_be_uint(n: uint);
    fn write_be_int(n: int);
    fn write_be_u64(n: u64);
    fn write_be_u32(n: u32);
    fn write_be_u16(n: u16);
    fn write_be_i64(n: i64);
    fn write_be_i32(n: i32);
    fn write_be_i16(n: i16);
    fn write_le_u64(n: u64);
    fn write_le_u32(n: u32);
    fn write_le_u16(n: u16);
    fn write_le_i64(n: i64);
    fn write_le_i32(n: i32);
    fn write_le_i16(n: i16);
    fn write_u8(n: u8);
}

impl<T: Writer> T : WriterUtil {
    fn write_char(ch: char) {
        if ch as uint < 128u {
            self.write(&[ch as u8]);
        } else {
            self.write_str(str::from_char(ch));
        }
    }
    fn write_str(s: &str) { str::byte_slice(s, |v| self.write(v)) }
    fn write_line(s: &str) {
        self.write_str(s);
        self.write_str(&"\n");
    }
    fn write_int(n: int) {
        int::to_str_bytes(n, 10u, |buf| self.write(buf))
    }
    fn write_uint(n: uint) {
        uint::to_str_bytes(false, n, 10u, |buf| self.write(buf))
    }
    fn write_le_uint(n: uint) {
        u64_to_le_bytes(n as u64, uint::bytes, |v| self.write(v))
    }
    fn write_le_int(n: int) {
        u64_to_le_bytes(n as u64, int::bytes, |v| self.write(v))
    }
    fn write_be_uint(n: uint) {
        u64_to_be_bytes(n as u64, uint::bytes, |v| self.write(v))
    }
    fn write_be_int(n: int) {
        u64_to_be_bytes(n as u64, int::bytes, |v| self.write(v))
    }
    fn write_be_u64(n: u64) {
        u64_to_be_bytes(n, 8u, |v| self.write(v))
    }
    fn write_be_u32(n: u32) {
        u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_be_u16(n: u16) {
        u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_be_i64(n: i64) {
        u64_to_be_bytes(n as u64, 8u, |v| self.write(v))
    }
    fn write_be_i32(n: i32) {
        u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_be_i16(n: i16) {
        u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_le_u64(n: u64) {
        u64_to_le_bytes(n, 8u, |v| self.write(v))
    }
    fn write_le_u32(n: u32) {
        u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_le_u16(n: u16) {
        u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }
    fn write_le_i64(n: i64) {
        u64_to_le_bytes(n as u64, 8u, |v| self.write(v))
    }
    fn write_le_i32(n: i32) {
        u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }
    fn write_le_i16(n: i16) {
        u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }

    fn write_u8(n: u8) { self.write(&[n]) }
}

fn file_writer(path: &Path, flags: ~[FileFlag]) -> Result<Writer, ~str> {
    result::chain(mk_file_writer(path, flags), |w| result::Ok(w))
}


// FIXME: fileflags // #2004
fn buffered_file_writer(path: &Path) -> Result<Writer, ~str> {
    let f = do os::as_c_charp(path.to_str()) |pathbuf| {
        do os::as_c_charp("w") |modebuf| {
            libc::fopen(pathbuf, modebuf)
        }
    };
    return if f as uint == 0u { result::Err(~"error opening "
                                            + path.to_str()) }
    else { result::Ok(FILE_writer(f, true)) }
}

// FIXME (#2004) it would be great if this could be a const
// FIXME (#2004) why are these different from the way stdin() is
// implemented?
fn stdout() -> Writer { fd_writer(libc::STDOUT_FILENO as c_int, false) }
fn stderr() -> Writer { fd_writer(libc::STDERR_FILENO as c_int, false) }

fn print(s: &str) { stdout().write_str(s); }
fn println(s: &str) { stdout().write_line(s); }

type MemBuffer = @{buf: DVec<u8>, mut pos: uint};

impl MemBuffer: Writer {
    fn write(v: &[const u8]) {
        // Fast path.
        let vlen = vec::len(v);
        let buf_len = self.buf.len();
        if self.pos == buf_len {
            self.buf.push_all(v);
            self.pos += vlen;
            return;
        }

        // FIXME #2004--use memcpy here?
        let mut pos = self.pos, vpos = 0u;
        while vpos < vlen && pos < buf_len {
            self.buf.set_elt(pos, copy v[vpos]);
            pos += 1u;
            vpos += 1u;
        }
        self.buf.push_slice(v, vpos, vlen);
        self.pos += vlen;
    }
    fn seek(offset: int, whence: SeekStyle) {
        let pos = self.pos;
        let len = self.buf.len();
        self.pos = seek_in_buf(offset, pos, len, whence);
    }
    fn tell() -> uint { self.pos }
    fn flush() -> int { 0 }
    fn get_type() -> WriterType { File }
}

fn mem_buffer() -> MemBuffer {
    @{buf: DVec(), mut pos: 0u}
}
fn mem_buffer_writer(b: MemBuffer) -> Writer { b as Writer }
fn mem_buffer_buf(b: MemBuffer) -> ~[u8] { b.buf.get() }
fn mem_buffer_str(b: MemBuffer) -> ~str {
    str::from_bytes(b.buf.get())
}

fn with_str_writer(f: fn(Writer)) -> ~str {
    let buf = mem_buffer();
    let wr = mem_buffer_writer(buf);
    f(wr);
    io::mem_buffer_str(buf)
}

fn with_buf_writer(f: fn(Writer)) -> ~[u8] {
    let buf = mem_buffer();
    let wr = mem_buffer_writer(buf);
    f(wr);
    io::mem_buffer_buf(buf)
}

// Utility functions
fn seek_in_buf(offset: int, pos: uint, len: uint, whence: SeekStyle) ->
   uint {
    let mut bpos = pos as int;
    let blen = len as int;
    match whence {
      SeekSet => bpos = offset,
      SeekCur => bpos += offset,
      SeekEnd => bpos = blen + offset
    }
    if bpos < 0 { bpos = 0; } else if bpos > blen { bpos = blen; }
    return bpos as uint;
}

fn read_whole_file_str(file: &Path) -> Result<~str, ~str> {
    result::chain(read_whole_file(file), |bytes| {
        if str::is_utf8(bytes) {
            result::Ok(str::from_bytes(bytes))
       } else {
           result::Err(file.to_str() + ~" is not UTF-8")
       }
    })
}

// FIXME (#2004): implement this in a low-level way. Going through the
// abstractions is pointless.
fn read_whole_file(file: &Path) -> Result<~[u8], ~str> {
    result::chain(file_reader(file), |rdr| {
        result::Ok(rdr.read_whole_stream())
    })
}

// fsync related

mod fsync {

    enum Level {
        // whatever fsync does on that platform
        FSync,

        // fdatasync on linux, similiar or more on other platforms
        FDataSync,

        // full fsync
        //
        // You must additionally sync the parent directory as well!
        FullFSync,
    }


    // Artifacts that need to fsync on destruction
    struct Res<t> {
        let arg: Arg<t>;
        new(-arg: Arg<t>) { self.arg <- arg; }
        drop {
          match self.arg.opt_level {
            option::None => (),
            option::Some(level) => {
              // fail hard if not succesful
              assert(self.arg.fsync_fn(self.arg.val, level) != -1);
            }
          }
        }
    }

    type Arg<t> = {
        val: t,
        opt_level: Option<Level>,
        fsync_fn: fn@(t, Level) -> int
    };

    // fsync file after executing blk
    // FIXME (#2004) find better way to create resources within lifetime of
    // outer res
    fn FILE_res_sync(&&file: FILERes, opt_level: Option<Level>,
                  blk: fn(&&Res<*libc::FILE>)) {
        blk(Res({
            val: file.f, opt_level: opt_level,
            fsync_fn: fn@(&&file: *libc::FILE, l: Level) -> int {
                return os::fsync_fd(libc::fileno(file), l) as int;
            }
        }));
    }

    // fsync fd after executing blk
    fn fd_res_sync(&&fd: FdRes, opt_level: Option<Level>,
                   blk: fn(&&Res<fd_t>)) {
        blk(Res({
            val: fd.fd, opt_level: opt_level,
            fsync_fn: fn@(&&fd: fd_t, l: Level) -> int {
                return os::fsync_fd(fd, l) as int;
            }
        }));
    }

    // Type of objects that may want to fsync
    trait FSyncable { fn fsync(l: Level) -> int; }

    // Call o.fsync after executing blk
    fn obj_sync(&&o: FSyncable, opt_level: Option<Level>,
                blk: fn(&&Res<FSyncable>)) {
        blk(Res({
            val: o, opt_level: opt_level,
            fsync_fn: fn@(&&o: FSyncable, l: Level) -> int {
                return o.fsync(l);
            }
        }));
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_simple() {
        let tmpfile = &Path("tmp/lib-io-test-simple.tmp");
        log(debug, tmpfile);
        let frood: ~str =
            ~"A hoopy frood who really knows where his towel is.";
        log(debug, frood);
        {
            let out: io::Writer =
                result::get(
                    io::file_writer(tmpfile, ~[io::Create, io::Truncate]));
            out.write_str(frood);
        }
        let inp: io::Reader = result::get(io::file_reader(tmpfile));
        let frood2: ~str = inp.read_c_str();
        log(debug, frood2);
        assert frood == frood2;
    }

    #[test]
    fn test_readchars_empty() {
        do io::with_str_reader(~"") |inp| {
            let res : ~[char] = inp.read_chars(128u);
            assert(vec::len(res) == 0u);
        }
    }

    #[test]
    fn test_readchars_wide() {
        let wide_test = ~"生锈的汤匙切肉汤hello生锈的汤匙切肉汤";
        let ivals : ~[int] = ~[
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748,
            104, 101, 108, 108, 111,
            29983, 38152, 30340, 27748,
            21273, 20999, 32905, 27748];
        fn check_read_ln(len : uint, s: ~str, ivals: ~[int]) {
            do io::with_str_reader(s) |inp| {
                let res : ~[char] = inp.read_chars(len);
                if (len <= vec::len(ivals)) {
                    assert(vec::len(res) == len);
                }
                assert(vec::slice(ivals, 0u, vec::len(res)) ==
                       vec::map(res, |x| x as int));
            }
        }
        let mut i = 0u;
        while i < 8u {
            check_read_ln(i, wide_test, ivals);
            i += 1u;
        }
        // check a long read for good measure
        check_read_ln(128u, wide_test, ivals);
    }

    #[test]
    fn test_readchar() {
        do io::with_str_reader(~"生") |inp| {
            let res : char = inp.read_char();
            assert(res as int == 29983);
        }
    }

    #[test]
    fn test_readchar_empty() {
        do io::with_str_reader(~"") |inp| {
            let res : char = inp.read_char();
            assert(res as int == -1);
        }
    }

    #[test]
    fn file_reader_not_exist() {
        match io::file_reader(&Path("not a file")) {
          result::Err(e) => {
            assert e == ~"error opening not a file";
          }
          result::Ok(_) => fail
        }
    }

    #[test]
    fn file_writer_bad_name() {
        match io::file_writer(&Path("?/?"), ~[]) {
          result::Err(e) => {
            assert str::starts_with(e, "error opening");
          }
          result::Ok(_) => fail
        }
    }

    #[test]
    fn buffered_file_writer_bad_name() {
        match io::buffered_file_writer(&Path("?/?")) {
          result::Err(e) => {
            assert str::starts_with(e, "error opening");
          }
          result::Ok(_) => fail
        }
    }

    #[test]
    fn mem_buffer_overwrite() {
        let mbuf = mem_buffer();
        mbuf.write(~[0u8, 1u8, 2u8, 3u8]);
        assert mem_buffer_buf(mbuf) == ~[0u8, 1u8, 2u8, 3u8];
        mbuf.seek(-2, SeekCur);
        mbuf.write(~[4u8, 5u8, 6u8, 7u8]);
        assert mem_buffer_buf(mbuf) == ~[0u8, 1u8, 4u8, 5u8, 6u8, 7u8];
        mbuf.seek(-2, SeekEnd);
        mbuf.write(~[8u8]);
        mbuf.seek(1, SeekSet);
        mbuf.write(~[9u8]);
        assert mem_buffer_buf(mbuf) == ~[0u8, 9u8, 4u8, 5u8, 8u8, 7u8];
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
