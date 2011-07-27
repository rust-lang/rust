
import str::sbuf;
import vec::vbuf;


// FIXME Somehow merge stuff duplicated here and macosx_os.rs. Made difficult
// by https://github.com/graydon/rust/issues#issue/268
native "cdecl" mod libc = "" {
    fn open(s: sbuf, flags: int, mode: uint) -> int;
    fn read(fd: int, buf: vbuf, count: uint) -> int;
    fn write(fd: int, buf: vbuf, count: uint) -> int;
    fn close(fd: int) -> int;
    type FILE;
    fn fopen(path: sbuf, mode: sbuf) -> FILE;
    fn fdopen(fd: int, mode: sbuf) -> FILE;
    fn fclose(f: FILE);
    fn fgetc(f: FILE) -> int;
    fn ungetc(c: int, f: FILE);
    fn feof(f: FILE) -> int;
    fn fread(buf: vbuf, size: uint, n: uint, f: FILE) -> uint;
    fn fwrite(buf: vbuf, size: uint, n: uint, f: FILE) -> uint;
    fn fseek(f: FILE, offset: int, whence: int) -> int;
    fn ftell(f: FILE) -> int;
    type dir;
    fn opendir(d: sbuf) -> dir;
    fn closedir(d: dir) -> int;
    type dirent;
    fn readdir(d: dir) -> dirent;
    fn getenv(n: sbuf) -> sbuf;
    fn setenv(n: sbuf, v: sbuf, overwrite: int) -> int;
    fn unsetenv(n: sbuf) -> int;
    fn pipe(buf: *mutable int) -> int;
    fn waitpid(pid: int, status: &mutable int, options: int) -> int;
}

native "cdecl" mod libc_ivec = "" {
    fn read(fd: int, buf: *u8, count: uint) -> int;
    fn write(fd: int, buf: *u8, count: uint) -> int;
    fn fread(buf: *u8, size: uint, n: uint, f: libc::FILE) -> uint;
    fn fwrite(buf: *u8, size: uint, n: uint, f: libc::FILE) -> uint;
}

mod libc_constants {
    fn O_RDONLY() -> int { ret 0; }
    fn O_WRONLY() -> int { ret 1; }
    fn O_RDWR() -> int { ret 2; }
    fn O_APPEND() -> int { ret 1024; }
    fn O_CREAT() -> int { ret 64; }
    fn O_EXCL() -> int { ret 128; }
    fn O_TRUNC() -> int { ret 512; }
    fn O_TEXT() -> int {
        ret 0; // nonexistent in linux libc

    }
    fn O_BINARY() -> int {
        ret 0; // nonexistent in linux libc

    }
    fn S_IRUSR() -> uint { ret 256u; }
    fn S_IWUSR() -> uint { ret 128u; }
}

fn exec_suffix() -> str { ret ""; }

fn target_os() -> str { ret "linux"; }

fn dylib_filename(base: str) -> str { ret "lib" + base + ".so"; }

fn pipe() -> {in: int, out: int} {
    let fds = {mutable in: 0, mutable out: 0};
    assert (os::libc::pipe(ptr::addr_of(fds.in)) == 0);
    ret {in: fds.in, out: fds.out};
}

fn fd_FILE(fd: int) -> libc::FILE { ret libc::fdopen(fd, str::buf("r")); }

fn waitpid(pid: int) -> int {
    let status = 0;
    assert (os::libc::waitpid(pid, status, 0) != -1);
    ret status;
}

native "rust" mod rustrt {
    fn rust_getcwd() -> str;
}

fn getcwd() -> str { ret rustrt::rust_getcwd(); }


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
