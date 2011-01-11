import std.os.libc;
import std._str;
import std._vec;


type stdio_reader = state obj {
                          fn getc() -> int;
                          fn ungetc(int i);
};

fn new_stdio_reader(str path) -> stdio_reader {
    state obj stdio_FILE_reader(os.libc.FILE f) {
        fn getc() -> int {
            ret os.libc.fgetc(f);
        }
        fn ungetc(int i) {
            os.libc.ungetc(i, f);
        }
        drop {
            os.libc.fclose(f);
        }
    }
    auto FILE = os.libc.fopen(_str.buf(path), _str.buf("r"));
    check (FILE as uint != 0u);
    ret stdio_FILE_reader(FILE);
}


type buf_reader = state obj {
                        fn read() -> vec[u8];
};

type buf_writer = state obj {
                        fn write(vec[u8] v);
};

fn default_bufsz() -> uint {
    ret 4096u;
}

fn new_buf() -> vec[u8] {
    ret _vec.alloc[u8](default_bufsz());
}

fn new_buf_reader(str path) -> buf_reader {

    state obj fd_buf_reader(int fd, mutable vec[u8] buf) {

        fn read() -> vec[u8] {

            // Ensure our buf is singly-referenced.
            if (_vec.rustrt.refcount[u8](buf) != 1u) {
                buf = new_buf();
            }

            auto len = default_bufsz();
            auto vbuf = _vec.buf[u8](buf);
            auto count = os.libc.read(fd, vbuf, len);

            if (count < 0) {
                log "error filling buffer";
                log sys.rustrt.last_os_error();
                fail;
            }

            _vec.len_set[u8](buf, count as uint);
            ret buf;
        }

        drop {
            os.libc.close(fd);
        }
    }

    auto fd = os.libc.open(_str.buf(path),
                           os.libc_constants.O_RDONLY() |
                           os.libc_constants.O_BINARY(),
                           0u);

    if (fd < 0) {
        log "error opening file for reading";
        log sys.rustrt.last_os_error();
        fail;
    }
    ret fd_buf_reader(fd, new_buf());
}

/**
 * FIXME (issue #150):  This should be
 *
 *   type fileflag = tag(append(), create(), truncate());
 *
 * but then the tag value ctors are not found from crate-importers of std, so
 * we manually simulate the enum below.
 */
type fileflag = uint;
fn append() -> uint { ret 0u; }
fn create() -> uint { ret 1u; }
fn truncate() -> uint { ret 2u; }

fn new_buf_writer(str path, vec[fileflag] flags) -> buf_writer {

    state obj fd_buf_writer(int fd) {

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

        drop {
            os.libc.close(fd);
        }
    }

    let int fflags =
        os.libc_constants.O_WRONLY() |
        os.libc_constants.O_BINARY();

    for (fileflag f in flags) {
        alt (f) {
            // FIXME (issue #150): cf comment above defn of fileflag type
            //case (append())   { fflags |= os.libc_constants.O_APPEND(); }
            //case (create())   { fflags |= os.libc_constants.O_CREAT(); }
            //case (truncate()) { fflags |= os.libc_constants.O_TRUNC(); }
            case (0u)   { fflags |= os.libc_constants.O_APPEND(); }
            case (1u)   { fflags |= os.libc_constants.O_CREAT(); }
            case (2u) { fflags |= os.libc_constants.O_TRUNC(); }
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
    ret fd_buf_writer(fd);
}

type writer =
    state obj {
          fn write_str(str s);
          fn write_int(int n);
          fn write_uint(uint n);
    };

fn file_writer(str path,
               vec[fileflag] flags)
    -> writer
{
    state obj fw(buf_writer out) {
        fn write_str(str s)   { out.write(_str.bytes(s)); }
        fn write_int(int n)   { out.write(_str.bytes(_int.to_str(n, 10u))); }
        fn write_uint(uint n) { out.write(_str.bytes(_uint.to_str(n, 10u))); }
    }
    ret fw(new_buf_writer(path, flags));
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
