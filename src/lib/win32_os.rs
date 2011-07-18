
import str::sbuf;
import vec::vbuf;

native "cdecl" mod libc = "" {
    fn open(sbuf s, int flags, uint mode) -> int = "_open";
    fn read(int fd, vbuf buf, uint count) -> int = "_read";
    fn write(int fd, vbuf buf, uint count) -> int = "_write";
    fn close(int fd) -> int = "_close";
    type FILE;
    fn fopen(sbuf path, sbuf mode) -> FILE;
    fn _fdopen(int fd, sbuf mode) -> FILE;
    fn fclose(FILE f);
    fn fgetc(FILE f) -> int;
    fn ungetc(int c, FILE f);
    fn feof(FILE f) -> int;
    fn fread(vbuf buf, uint size, uint n, FILE f) -> uint;
    fn fwrite(vbuf buf, uint size, uint n, FILE f) -> uint;
    fn fseek(FILE f, int offset, int whence) -> int;
    fn ftell(FILE f) -> int;
    fn _pipe(*mutable int fds, uint size, int mode) -> int;
}

native "cdecl" mod libc_ivec = "" {
    fn read(int fd, *u8 buf, uint count) -> int;
    fn write(int fd, *u8 buf, uint count) -> int;
    fn fread(*u8 buf, uint size, uint n, libc::FILE f) -> uint;
    fn fwrite(*u8 buf, uint size, uint n, libc::FILE f) -> uint;
}

mod libc_constants {
    fn O_RDONLY() -> int { ret 0; }
    fn O_WRONLY() -> int { ret 1; }
    fn O_RDWR() -> int { ret 2; }
    fn O_APPEND() -> int { ret 1024; }
    fn O_CREAT() -> int { ret 64; }
    fn O_EXCL() -> int { ret 128; }
    fn O_TRUNC() -> int { ret 512; }
    fn O_TEXT() -> int { ret 16384; }
    fn O_BINARY() -> int { ret 32768; }
    fn S_IRUSR() -> uint {
        ret 256u; // really _S_IREAD  in win32

    }
    fn S_IWUSR() -> uint {
        ret 128u; // really _S_IWRITE in win32

    }
}

fn exec_suffix() -> str { ret ".exe"; }

fn target_os() -> str { ret "win32"; }

fn dylib_filename(str base) -> str { ret base + ".dll"; }

fn pipe() -> tup(int, int) {
    auto fds = tup(mutable 0, 0);
    assert (os::libc::_pipe(ptr::addr_of(fds._0), 1024u,
                            libc_constants::O_BINARY()) == 0);
    ret tup(fds._0, fds._1);
}

fn fd_FILE(int fd) -> libc::FILE { ret libc::_fdopen(fd, str::buf("r")); }

native "rust" mod rustrt {
    fn rust_process_wait(int handle) -> int;
    fn rust_getcwd() -> str;
    fn rust_SetEnvironmentVariable(sbuf n, sbuf v) -> int;
    fn rust_GetEnvironmentVariable(sbuf n, sbuf v, uint nsize) -> uint;
}

fn waitpid(int pid) -> int { ret rustrt::rust_process_wait(pid); }

fn getcwd() -> str { ret rustrt::rust_getcwd(); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
